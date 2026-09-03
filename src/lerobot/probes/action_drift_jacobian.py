#!/usr/bin/env python
r"""
Subtask-conditioned action sensitivity.

For a real validation frame, let $E_k$ be one multimodal conditioning token and
let $V_G$ be the continuous flow output restricted to one robot action group.
This probe estimates the raw local Jacobian-block norm

$$S_{f,k}^{G}=\left\|\frac{\partial\,\mathrm{vec}(V_G)}{\partial E_k}\right\|_F.$$

The three groups are fixed by the reBot action schema: proximal arm
(shoulder pan/lift + elbow flex), wrist (flex/yaw/roll), and gripper. Scores are
never normalised across cameras, prompt, or action groups. Comparisons are made
only for the same group and same input segment across dataset subtasks.

The conditioning frame, prompt, history, metadata, and GT action chunk all come
from the validation dataset. The GT chunk anchors the standard teacher-forced
flow state at ``probe_parameters.timestep``; fixed model-native flow noise is
recorded per frame. A small number of Hutchinson vector-Jacobian products makes
the full-horizon Frobenius norm practical without constructing the full Jacobian.

How to read the pictures
------------------------

A bright patch means *this frame's flow velocity for this action group moves more when
that patch's token is perturbed*. It does not mean the model "looked at" the patch —
attention is a different probe — and it is not a claim about correctness.

Brightness is always relative to a chosen scale, never absolute, and that choice is
the difference between a picture and a wash:

* ``fixed · run p99`` (the default) — the 99th percentile of every token value in the
  run. Frames stay comparable to each other and a typical frame still uses most of the
  ramp. The loudest patches clip; that is the trade.
* ``fixed · run max`` — the old behaviour, kept because it is the only mode where
  nothing clips. One outlying frame sets the ramp for all of them, so most frames look
  flat: on a representative run the wrist camera's peak frame is ~50x the median
  frame's peak, leaving typical frames painting in the bottom 2% of the ramp.
* ``auto · shown frames`` — spans only the frames on screen. Full contrast, and in the
  two-frame comparison both frames still share one ramp, so A really is brighter than B
  when it is larger. Values are not comparable to a frame that is not on screen.

The contrast control applies $x^{\gamma}$ to the normalised value before colouring;
these scores span decades, so a linear ramp hides structure that is really there. The
default picks $\gamma$ from the data rather than guessing it: with $m$ the median of the
normalised values actually on screen, $\gamma=\ln(1/2)/\ln m$ puts that median at
mid-ramp, so an already-spread panel gets $\gamma\approx1$ and a spiky one gets the
compression it needs. Fixed sqrt/linear/cube-root remain selectable. Every panel carries
its own colourbar with raw tick values, and hovering a patch reads out its raw value.

Comparisons that are valid: same action group and same input segment, across subtasks
or across frames. Comparisons that are not: one camera against another, camera against
prompt, or one action group against another — those scales are deliberately never
unified, because the quantities have different units of "one token".

Outputs under ``action_drift_jacobian/``:

* ``sensitivity.html`` — frame browser; overlays are drawn in the browser from the
  patch grids, so scale, contrast, and two-frame comparison need no re-run.
* ``prompt_<group>.png`` — episode-balanced prompt-token sensitivity per subtask, with a
  caption stating the quantity, the averaging, and which direction is comparable.
* ``frame_metrics.csv`` / ``summary.json`` — raw aggregate values and provenance.
* ``raw/*.npz`` — per frame, the flat token score vector per action group plus each
  camera's patch grid as ``<group>__<camera>_grid``, already shaped rows x cols.
"""

import csv
import json
import logging
import os
import random
import re
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import lerobot.rl.molmoact2.rl_molmoact2  # noqa: F401 — register MolmoAct2RLConfig for CLI
from lerobot.cameras import opencv, realsense  # noqa: F401 — CLI config registry
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ActionSensitivityResult, AttentionCaptureResult, ProbablePolicy
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    build_episode_index,
    get_subtask_idx,
    get_subtask_str,
    load_extra_dataset,
    load_probe_dataset,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
)
from lerobot.robots import rebot_b601_follower, so_follower  # noqa: F401 — CLI config registry
from lerobot.teleoperators import rebot_102_leader, so_leader  # noqa: F401 — CLI config registry
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

ACTION_GROUPS: dict[str, list[int]] = {
    "proximal_arm": [0, 1, 2],
    "wrist": [3, 4, 5],
    "gripper": [6],
}

GROUP_LABELS = {
    "proximal_arm": "Proximal arm",
    "wrist": "Wrist",
    "gripper": "Gripper",
}


@dataclass
class ProbeJacobianConfig(TrainRLServerPipelineConfig):
    """Tunables live under ``cfg.probe_parameters``."""


def _caption(text: str, fig_width: float) -> str:
    """Hard-wrap a caption to the figure: ~16 characters per inch at 8pt."""
    return textwrap.fill(" ".join(text.split()), width=max(60, int(fig_width * 16)))


def _slug(value: str, limit: int = 72) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return (clean or "unlabelled")[:limit]


def _evenly_spaced(items: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count <= 0 or len(items) <= count:
        return list(items)
    positions = np.linspace(0, len(items) - 1, count)
    return [items[int(round(pos))] for pos in positions]


def build_subtask_samples(dataset, cfg) -> list[dict[str, Any]]:
    """Sample real frames evenly inside every episode/subtask intersection."""
    p = cfg.probe_parameters
    ep_to_indices = build_episode_index(dataset)
    selected: list[int] = []
    explicit = getattr(p, "attn_eval_episodes", None)
    if explicit:
        selected.extend(
            ep for ep in (int(value) for value in str(explicit).split(",")) if ep in ep_to_indices
        )
    all_eps = list(ep_to_indices)
    rng = random.Random(int(p.random_seed))
    rng.shuffle(all_eps)
    max_episodes = int(p.max_episodes) if p.max_episodes is not None else len(all_eps)
    for episode in all_eps:
        if len(selected) >= max_episodes:
            break
        if episode not in selected:
            selected.append(episode)
    selected = sorted(selected[:max_episodes])

    grid_stride = max(1, int(probe_image_stride(cfg)))
    requested_per_subtask = max(
        1, int(getattr(p, "action_sensitivity_frames_per_subtask", 6))
    )
    max_per_episode = max(1, int(p.n_frames_per_episode))
    out: list[dict[str, Any]] = []
    for episode in selected:
        buckets: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for frame_idx, global_idx in enumerate(ep_to_indices[episode]):
            if frame_idx % grid_stride:
                continue
            subtask_idx = get_subtask_idx(dataset, global_idx)
            subtask = get_subtask_str(dataset, subtask_idx) or "(unlabelled)"
            buckets[subtask].append((frame_idx, global_idx))

        if not buckets:
            continue
        per_subtask = min(
            requested_per_subtask,
            max(1, max_per_episode // len(buckets)),
        )
        episode_rows: list[dict[str, Any]] = []
        for subtask, candidates in sorted(buckets.items()):
            for frame_idx, global_idx in _evenly_spaced(candidates, per_subtask):
                episode_rows.append(
                    {
                        "episode": int(episode),
                        "frame": int(frame_idx),
                        "global_idx": int(global_idx),
                        "subtask": subtask,
                    }
                )
        episode_rows.sort(key=lambda row: row["frame"])
        out.extend(episode_rows[:max_per_episode])
    return out


def _score_view(capture: ActionSensitivityResult, group: str) -> AttentionCaptureResult:
    scores = capture.scores_by_group[group].detach().float().cpu()
    return replace(
        capture.token_metadata,
        cross_attn_by_layer={0: scores.reshape(1, 1, 1, -1)},
    )


def _prompt_entries(result: AttentionCaptureResult) -> list[dict[str, Any]]:
    from lerobot.probes.attention import _group_prompt_indices, _text_blocks_for_action_matrix

    cross = result.cross_attn_by_layer[0][0, 0, 0].detach().float().cpu()
    text_blocks = _text_blocks_for_action_matrix(result)
    encoder_valid = None
    if torch.is_tensor(result.encoder_pad_masks) and result.encoder_pad_masks.ndim >= 2:
        encoder_valid = result.encoder_pad_masks[0].detach().cpu().to(torch.bool)
    grouped = _group_prompt_indices(result, text_blocks, int(cross.shape[0]), encoder_valid)
    entries: list[dict[str, Any]] = []
    for clause, tokens in grouped:
        for token_idx, label in tokens:
            entries.append(
                {
                    "index": int(token_idx),
                    "clause": str(clause),
                    "label": str(label).strip() or "·",
                    "value": float(cross[int(token_idx)].item()),
                }
            )
    return entries


def _capture_records(adapter, dataset, cfg, samples) -> list[dict[str, Any]]:
    from lerobot.probes.attention import _extract_overlay_grids

    p = cfg.probe_parameters
    timestep = float(getattr(p, "timestep", 0.5))
    projections = max(1, int(getattr(p, "action_sensitivity_projections", 4)))
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        logging.info(
            "[action_sensitivity] %d/%d ep%d fr%d — %s",
            index,
            len(samples),
            sample["episode"],
            sample["frame"],
            sample["subtask"],
        )
        frame = probe_frame_inputs(dataset, cfg, sample["global_idx"], adapter.chunk_size)
        capture = adapter.capture_action_sensitivity(
            frame["obs"],
            frame["task"],
            gt_actions=frame["gt_actions"],
            action_groups=ACTION_GROUPS,
            timestep=timestep,
            num_projections=projections,
            seed=int(p.random_seed) + 1009 * int(sample["global_idx"]),
            subtask=frame["subtask"],
            metadata=frame["metadata"],
        )
        group_data: dict[str, dict[str, Any]] = {}
        for group in ACTION_GROUPS:
            view = _score_view(capture, group)
            grids = {}
            grid_images = {}
            for grid in _extract_overlay_grids(view, 0):
                name = str(grid["cam_name"])
                rows, cols = (int(value) for value in grid["grid_hw"])
                grids[name] = grid["per_head_grid"].squeeze(0).numpy().reshape(rows, cols)
                grid_images[name] = grid
            group_data[group] = {
                "grids": grids,
                "grid_images": grid_images,
                "prompt": _prompt_entries(view),
                "scores": capture.scores_by_group[group].detach().float().cpu().numpy(),
            }
        records.append({**sample, "capture": capture, "groups": group_data})
    return records


def _global_scales(records) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, float]]]:
    """Fixed display scales per (group, camera) and per group's prompt.

    Both a ``p99`` over every token value in the run and the outright ``max``. The max
    alone is a bad default: one loud frame sets the ramp for all of them, and with the
    wrist camera's peak frame at ~50x the median frame's peak, a typical frame paints
    in the bottom few percent of the ramp and reads as empty when it is not.
    """
    camera_values: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    prompt_values: dict[str, list[float]] = {group: [] for group in ACTION_GROUPS}
    for record in records:
        for group, data in record["groups"].items():
            for camera, grid in data["grids"].items():
                camera_values[(group, camera)].append(np.asarray(grid, dtype=np.float64).reshape(-1))
            prompt_values[group].extend(float(token["value"]) for token in data["prompt"])

    def scale(values: np.ndarray) -> dict[str, float]:
        return {
            "max": max(float(values.max()), 1e-12),
            "p99": max(float(np.percentile(values, 99)), 1e-12),
        }

    camera_scales = {key: scale(np.concatenate(parts)) for key, parts in camera_values.items()}
    prompt_scales = {
        group: scale(np.asarray(values or [0.0], dtype=np.float64))
        for group, values in prompt_values.items()
    }
    return camera_scales, prompt_scales


def _write_details(output_dir, records) -> list[dict[str, Any]]:
    """One plain camera image per frame, plus the patch grids the page draws over it.

    The overlay used to be burned into a JPEG at a colour scale chosen here, so every
    display decision — the scale, the gamma, putting two frames on one ramp — cost a
    full re-run of the probe on a GPU. A grid is a few hundred floats, so it ships to
    the page instead and all of that becomes a browser-side choice.
    """
    from PIL import Image

    detail_root = os.path.join(output_dir, "details")
    raw_root = os.path.join(output_dir, "raw")
    os.makedirs(detail_root, exist_ok=True)
    os.makedirs(raw_root, exist_ok=True)
    html_records: list[dict[str, Any]] = []

    for record in records:
        stem = f"ep{record['episode']:04d}_fr{record['frame']:06d}"
        frame_dir = os.path.join(detail_root, stem)
        os.makedirs(frame_dir, exist_ok=True)
        raw_rel = os.path.join("raw", f"{stem}.npz")

        arrays = {group: data["scores"] for group, data in record["groups"].items()}
        cameras: dict[str, Any] = {}
        groups_json: dict[str, Any] = {}
        for group, data in record["groups"].items():
            grids_json = {}
            for camera, grid_info in data["grid_images"].items():
                rows, cols = (int(value) for value in grid_info["grid_hw"])
                values = np.asarray(data["grids"][camera], dtype=np.float64).reshape(-1)
                arrays[f"{group}__{camera}_grid"] = values.reshape(rows, cols)
                grids_json[camera] = [float(f"{value:.5g}") for value in values]
                if camera not in cameras:
                    filename = f"cam_{_slug(camera)}.jpg"
                    absolute = os.path.join(frame_dir, filename)
                    Image.fromarray(grid_info["img_np"]).save(absolute, quality=90)
                    cameras[camera] = {
                        "file": os.path.relpath(absolute, output_dir),
                        "hw": [rows, cols],
                    }
            groups_json[group] = {"grids": grids_json, "prompt": data["prompt"]}
        np.savez_compressed(os.path.join(output_dir, raw_rel), **arrays)

        html_records.append(
            {
                "episode": record["episode"],
                "frame": record["frame"],
                "global_idx": record["global_idx"],
                "subtask": record["subtask"],
                "raw": raw_rel,
                "cameras": cameras,
                "groups": groups_json,
            }
        )
    return html_records


def _aggregate_prompt(records, subtask: str, group: str) -> dict[tuple[str, str], float]:
    by_episode: dict[int, list[dict[tuple[str, str], float]]] = defaultdict(list)
    for record in records:
        if record["subtask"] != subtask:
            continue
        values: dict[tuple[str, str], float] = defaultdict(float)
        for token in record["groups"][group]["prompt"]:
            values[(token["clause"], token["label"])] += float(token["value"])
        by_episode[int(record["episode"])].append(dict(values))
    keys = set().union(*(set(frame) for frames in by_episode.values() for frame in frames))
    if not keys:
        return {}
    episode_means = []
    for frames in by_episode.values():
        episode_means.append(
            {key: float(np.mean([frame.get(key, 0.0) for frame in frames])) for key in keys}
        )
    return {
        key: float(np.mean([episode.get(key, 0.0) for episode in episode_means]))
        for key in keys
    }


def _plot_prompt_maps(output_dir, records) -> list[str]:
    subtasks = sorted({record["subtask"] for record in records})
    written = []
    for group in ACTION_GROUPS:
        fig, axes = plt.subplots(
            len(subtasks), 1,
            figsize=(9.0, max(2.5, 2.6 * len(subtasks)) + 1.2),
            squeeze=False,
            layout="constrained",
        )
        for row, subtask in enumerate(subtasks):
            ax = axes[row, 0]
            values = _aggregate_prompt(records, subtask, group)
            top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:12][::-1]
            if not top:
                ax.text(0.5, 0.5, "no prompt tokens", ha="center", va="center")
                ax.axis("off")
                continue
            positions = np.arange(len(top))
            ax.barh(positions, [value for _, value in top], color="#7c3aed")
            ax.set_yticks(positions)
            ax.set_yticklabels(
                [f"{clause} · {label}" for (clause, label), _ in top], fontsize=7
            )
            ax.set_title(subtask, fontsize=9, loc="left")
            ax.set_xlabel("raw token Jacobian norm (episode-balanced mean)", fontsize=8)
            ax.grid(axis="x", alpha=0.2)
        label = GROUP_LABELS[group].lower()
        fig.suptitle(f"{GROUP_LABELS[group]}: which prompt words move this joint group", fontsize=12)
        fig.supxlabel(
            _caption(
                f"""A bar is the Frobenius norm of d(flow velocity of the {label} joints) /
                d(prompt token embedding), summed over repeats of the same word inside a clause,
                averaged within an episode and then across episodes. Top 12 tokens per subtask, so
                a token missing from one row is only outside its top 12. Raw units on a linear
                axis: compare bars inside this figure, never against a camera panel or another
                action group.""",
                9.0,
            ),
            fontsize=7.5,
            color="#3f3f46",
        )
        filename = f"prompt_{group}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=145)
        plt.close(fig)
        written.append(filename)
    return written


def _write_frame_metrics(output_dir, records) -> None:
    with open(os.path.join(output_dir, "frame_metrics.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["episode", "frame", "global_idx", "subtask", "action_group", "input_segment",
             "token_mean", "token_max", "token_rms", "n_tokens"]
        )
        for record in records:
            for group, data in record["groups"].items():
                for segment, grid in data["grids"].items():
                    values = np.asarray(grid, dtype=np.float64).reshape(-1)
                    writer.writerow(
                        [record["episode"], record["frame"], record["global_idx"], record["subtask"],
                         group, segment, float(values.mean()), float(values.max()),
                         float(np.sqrt(np.mean(values ** 2))), int(values.size)]
                    )
                prompt = np.asarray([token["value"] for token in data["prompt"]], dtype=np.float64)
                if prompt.size:
                    writer.writerow(
                        [record["episode"], record["frame"], record["global_idx"], record["subtask"],
                         group, "prompt", float(prompt.mean()), float(prompt.max()),
                         float(np.sqrt(np.mean(prompt ** 2))), int(prompt.size)]
                    )


_HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subtask action sensitivity</title>
<style>
:root{--bg:#f5f5f1;--card:#fff;--ink:#18181b;--muted:#71717a;--line:#deded8;--accent:#6d28d9}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}
header{padding:22px 28px 14px;border-bottom:1px solid var(--line);background:#fafaf8}
h1{font-size:22px;margin:0 0 5px} .dek{color:var(--muted);max-width:1000px}
.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;padding:14px 28px;position:sticky;top:0;background:rgba(245,245,241,.96);backdrop-filter:blur(8px);z-index:5;border-bottom:1px solid var(--line)}
label{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
select,input{width:100%;margin-top:4px;padding:8px;border:1px solid var(--line);border-radius:7px;background:white}
input[type=range]{padding:0}
.switch{display:flex;align-items:center;gap:7px;margin-top:20px;font-size:13px;text-transform:none;letter-spacing:0;color:var(--ink)}
.switch input{width:auto;margin:0}
main{padding:20px 28px 40px;max-width:1750px;margin:auto}
.meta{display:flex;gap:18px;align-items:baseline;margin-bottom:14px;flex-wrap:wrap} .subtask{font-size:18px;font-weight:700}
.cbar{margin-top:9px}
.cbar .ramp{height:9px;border-radius:5px;border:1px solid var(--line)}
.cbar .ticks{position:relative;height:15px;font-size:10px;color:var(--muted)}
.cbar .ticks span{position:absolute;top:2px;white-space:nowrap}
.cbar .cnote{font-size:11px;color:var(--muted)}
.camera-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}
.camera-grid.wide{grid-template-columns:1fr}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;box-shadow:0 1px 2px #00000008}
.cells{display:grid;gap:10px} .cells.two{grid-template-columns:1fr 1fr}
.cell figure{margin:0} .pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.pair img,.pair canvas{width:100%;display:block;border-radius:5px;background:#111}
.cell .who{font-size:11px;color:var(--muted);margin-bottom:4px;display:flex;justify-content:space-between}
.cell .who b{color:var(--ink)}
.caption{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:7px}
.prompt{margin-top:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.tokens{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px} .token{border:1px solid #ddd6fe;border-radius:5px;padding:4px 6px;background:#faf5ff}
.token small{display:block;color:#6b21a8;font-size:9px}
.rowlabel{font-size:11px;color:var(--muted);margin-top:11px;text-transform:uppercase;letter-spacing:.07em}
.empty{color:var(--muted);padding:30px;text-align:center} .raw{margin-left:auto}
@media(max-width:850px){.camera-grid{grid-template-columns:1fr}.cells.two{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Subtask-conditioned action sensitivity</h1>
<div class="dek">One patch is one input token. Its value is the Frobenius norm of d(flow velocity of the selected joint group) / d(that token's embedding) at this real validation frame — how far the predicted motion moves when that patch's embedding is nudged. It is not attention and it is not correctness. Brightness is always relative to the scale under each panel, never absolute: a dark patch is small <i>compared to that scale</i>, not unimportant. Hover a patch for its raw value. Compare the same action group and the same input panel across frames or subtasks; camera, prompt and action-group scales are deliberately never unified.</div></header>
<div class="controls">
  <div><label>Subtask<select id="subtask"></select></label></div>
  <div><label>Action group<select id="group"></select></label></div>
  <div><label>Episode<select id="episode"></select></label></div>
  <div><label>Frame A <span id="frameCount"></span><input id="frame" type="range" min="0" value="0"></label></div>
  <div><label class="switch"><input id="compare" type="checkbox">Compare two frames</label></div>
  <div><label>Frame B<input id="frameB" type="range" min="0" value="0" disabled></label></div>
  <div><label>Colour scale<select id="mode">
    <option value="p99">fixed · run p99</option>
    <option value="max">fixed · run max</option>
    <option value="auto">auto · shown frames</option>
  </select></label></div>
  <div><label>Contrast<select id="gamma">
    <option value="auto">auto · median mid-ramp</option>
    <option value="0.5">fixed sqrt</option>
    <option value="1">fixed linear</option>
    <option value="0.33">fixed cube root</option>
  </select></label></div>
</div>
<main><div class="meta"><span class="subtask" id="title"></span><span id="where"></span><a class="raw" id="raw">raw NPZ</a></div>
<div class="camera-grid" id="cameras"></div>
<section class="prompt"><strong>Prompt-token sensitivity</strong><div id="promptRows"></div><div id="promptBar"></div></section></main>
<script>const DATA=__DATA__;
const $=id=>document.getElementById(id);
const sub=$('subtask'), group=$('group'), episode=$('episode'), slider=$('frame'), sliderB=$('frameB'),
      compare=$('compare'), mode=$('mode'), gammaSel=$('gamma');
const subtasks=[...new Set(DATA.records.map(r=>r.subtask))].sort();
sub.innerHTML=subtasks.map(s=>`<option>${s}</option>`).join('');
group.innerHTML=DATA.groups.map(g=>`<option value="${g}">${DATA.groupLabels[g]}</option>`).join('');

// Inferno control points: transparent-dark through purple and orange to yellow. Alpha
// rises with the value so low-sensitivity patches leave the photograph readable.
const RAMP=[[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],[245,125,21],[252,193,60],[252,255,164]];
function ramp(t){const x=Math.max(0,Math.min(1,t))*(RAMP.length-1), i=Math.min(RAMP.length-2,Math.floor(x)), f=x-i;
  return [0,1,2].map(c=>Math.round(RAMP[i][c]+(RAMP[i+1][c]-RAMP[i][c])*f));}
function rampCSS(){return `linear-gradient(90deg,${RAMP.map((c,i)=>`rgb(${c[0]},${c[1]},${c[2]}) ${i/(RAMP.length-1)*100}%`).join(',')})`;}

function episodes(){const vals=[...new Set(DATA.records.filter(r=>r.subtask===sub.value).map(r=>r.episode))].sort((a,b)=>a-b);
  episode.innerHTML='<option value="all">all episodes</option>'+vals.map(e=>`<option>${e}</option>`).join('');}
function filtered(){return DATA.records.filter(r=>r.subtask===sub.value&&(episode.value==='all'||String(r.episode)===episode.value));}
function shown(rows){const a=rows[+slider.value]; if(!compare.checked) return [a];
  return [a, rows[Math.min(+sliderB.value, rows.length-1)]];}

// One ramp for everything on screen. "auto" spans only the frames being looked at,
// which is what makes an A/B comparison honest and still uses the whole ramp.
function scaleFor(store, key, frames, pick){
  if(mode.value!=='auto') return store[key][mode.value];
  return Math.max(1e-12, ...frames.map(pick));
}

// Colour is (v/vmax)^gamma. These norms span decades, so one hardcoded exponent is a
// guess that fits some panels and washes out others. "auto" solves m^gamma = 0.5 for the
// median m of the values actually drawn: an already-spread panel gets gamma ~ 1, a panel
// carried by a few loud cells gets the compression it needs, and the numbers don't move.
function gammaFor(values){
  if(gammaSel.value!=='auto') return +gammaSel.value;
  const x=values.filter(v=>v>0&&v<1).sort((a,b)=>a-b);
  if(!x.length) return 1;
  const median=x[x.length>>1];
  return Math.min(4,Math.max(0.15,Math.log(0.5)/Math.log(median)));
}
// Ticks sit evenly along the colour ramp and are labelled with the raw value that colour
// stands for, so the gamma shows up as their uneven spacing in value.
function colourbar(vmax, gamma, what){
  const exp=Math.floor(Math.log10(vmax)), unit=Math.pow(10,exp);
  const ticks=[0,.25,.5,.75,1].map(t=>{
    const shift=t===0?'0':t===1?'-100%':'-50%';
    return `<span style="left:${t*100}%;transform:translateX(${shift})">${(vmax*Math.pow(t,1/gamma)/unit).toFixed(2)}</span>`;
  }).join('');
  return `<div class="cbar"><div class="ramp" style="background:${rampCSS()}"></div><div class="ticks">${ticks}</div>
    <div class="cnote">${what} &times; 1e${exp} &middot; full scale ${vmax.toExponential(2)} &middot; ${mode.options[mode.selectedIndex].text} &middot; &gamma; ${gamma.toFixed(2)}</div></div>`;
}
function drawCell(canvas, img, grid, hw, vmax, gamma){
  const w=canvas.width=img.naturalWidth||448, h=canvas.height=img.naturalHeight||448, ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,w,h); ctx.drawImage(img,0,0,w,h);
  const [rows,cols]=hw, off=document.createElement('canvas'); off.width=cols; off.height=rows;
  const octx=off.getContext('2d'), buf=octx.createImageData(cols,rows);
  for(let i=0;i<rows*cols;i++){
    const t=Math.pow(Math.max(0,Math.min(1,grid[i]/vmax)),gamma), [r,g,b]=ramp(t);
    buf.data[4*i]=r; buf.data[4*i+1]=g; buf.data[4*i+2]=b; buf.data[4*i+3]=Math.round(255*(0.10+0.80*t));
  }
  octx.putImageData(buf,0,0);
  ctx.imageSmoothingEnabled=true; ctx.imageSmoothingQuality='high';
  ctx.drawImage(off,0,0,w,h);
}
function cellHTML(rec, tag, camera){
  const cam=rec.cameras[camera];
  return `<div class="cell"><div class="who"><b>${tag} · ep${rec.episode} fr${rec.frame}</b><span class="val">${rec.subtask}</span></div>
    <div class="pair"><img src="${cam.file}" alt="${camera}"><canvas data-frame="${rec.global_idx}" data-cam="${camera}"></canvas></div></div>`;
}
function render(){
  const rows=filtered(), g=group.value;
  slider.max=sliderB.max=Math.max(0,rows.length-1);
  slider.value=Math.min(+slider.value,+slider.max); sliderB.value=Math.min(+sliderB.value,+sliderB.max);
  sliderB.disabled=!compare.checked;
  $('frameCount').textContent=`${rows.length} sampled`;
  if(!rows.length){$('cameras').innerHTML='<div class="empty">No sampled frames.</div>';$('promptRows').innerHTML='';$('promptBar').innerHTML='';return;}
  const frames=shown(rows), r=frames[0];
  $('title').textContent=r.subtask;
  $('where').textContent=frames.map(f=>`ep ${f.episode} · frame ${f.frame}`).join('   vs   ');
  $('raw').href=r.raw;

  const cameras=Object.keys(r.cameras);
  $('cameras').className='camera-grid'+(compare.checked?' wide':'');
  const vmaxByCam={}, gammaByCam={};
  cameras.forEach(camera=>{
    vmaxByCam[camera]=scaleFor(DATA.cameraScales, `${g}|${camera}`, frames,
      f=>Math.max(...f.groups[g].grids[camera]));
    gammaByCam[camera]=gammaFor([].concat(...frames.map(f=>f.groups[g].grids[camera])).map(v=>v/vmaxByCam[camera]));
  });
  $('cameras').innerHTML=cameras.map(camera=>
    `<article class="panel"><div class="cells ${compare.checked?'two':''}">${frames.map((f,i)=>cellHTML(f,i?'B':'A',camera)).join('')}</div>
     <div class="caption"><b>${camera}</b></div>${colourbar(vmaxByCam[camera],gammaByCam[camera],'raw Jacobian norm')}</article>`
  ).join('')||'<div class="empty">No camera patches mapped.</div>';
  $('cameras').querySelectorAll('canvas').forEach(canvas=>{
    const camera=canvas.dataset.cam, frame=frames.find(f=>String(f.global_idx)===canvas.dataset.frame);
    const grid=frame.groups[g].grids[camera], hw=frame.cameras[camera].hw;
    const img=canvas.previousElementSibling;
    const paint=()=>drawCell(canvas,img,grid,hw,vmaxByCam[camera],gammaByCam[camera]);
    if(img.complete&&img.naturalWidth) paint(); else img.onload=paint;
    // Colour answers "how does this patch compare"; the readout answers "how much".
    const readout=canvas.closest('.cell').querySelector('.val'), subtask=frame.subtask;
    canvas.onmousemove=event=>{
      const box=canvas.getBoundingClientRect(), [gridRows,gridCols]=hw;
      const col=Math.min(gridCols-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*gridCols)));
      const row=Math.min(gridRows-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*gridRows)));
      readout.textContent=`patch r${row} c${col} · ${grid[row*gridCols+col].toExponential(3)}`;
    };
    canvas.onmouseleave=()=>{readout.textContent=subtask};
  });

  const pmax=scaleFor(DATA.promptScales, g, frames, f=>Math.max(...f.groups[g].prompt.map(t=>t.value)));
  const pgamma=gammaFor([].concat(...frames.map(f=>f.groups[g].prompt.map(t=>t.value/pmax))));
  $('promptBar').innerHTML=colourbar(pmax,pgamma,'raw Jacobian norm');
  $('promptRows').innerHTML=frames.map((f,i)=>{
    const tokens=f.groups[g].prompt.map(t=>{
      const x=Math.pow(Math.max(0,Math.min(1,t.value/pmax)),pgamma), [r0,g0,b0]=ramp(x);
      return `<span class="token" style="background:rgba(${r0},${g0},${b0},${(0.10+0.75*x).toFixed(3)})" title="raw ${t.value.toExponential(4)}"><small>${t.clause}</small>${t.label}</span>`;
    }).join('')||'<span class="empty">No prompt tokens mapped.</span>';
    return `${compare.checked?`<div class="rowlabel">${i?'B':'A'} · ep${f.episode} fr${f.frame}</div>`:''}<div class="tokens">${tokens}</div>`;
  }).join('');
}
sub.onchange=()=>{episodes();slider.value=0;sliderB.value=0;render()};
episode.onchange=()=>{slider.value=0;sliderB.value=0;render()};
[group,mode,gammaSel,compare].forEach(el=>el.onchange=render);
[slider,sliderB].forEach(el=>el.oninput=render);
episodes();render();
</script></body></html>"""


def _write_html(output_dir, html_records, camera_scales, prompt_scales) -> str:
    payload = {
        "records": html_records,
        "groups": list(ACTION_GROUPS),
        "groupLabels": GROUP_LABELS,
        "cameraScales": {f"{group}|{camera}": value for (group, camera), value in camera_scales.items()},
        "promptScales": prompt_scales,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    document = _HTML_TEMPLATE.replace("__DATA__", data_json)
    path = os.path.join(output_dir, "sensitivity.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(document)
    return "sensitivity.html"




def _summary(records, camera_scales, projections, timestep) -> dict[str, Any]:
    subtasks = sorted({record["subtask"] for record in records})
    episodes = sorted({int(record["episode"]) for record in records})
    counts: dict[str, dict[str, int]] = {}
    for subtask in subtasks:
        subset = [record for record in records if record["subtask"] == subtask]
        counts[subtask] = {
            "frames": len(subset),
            "episodes": len({record["episode"] for record in subset}),
        }
    return {
        "metric": "raw Frobenius norm of d(grouped flow velocity)/d(multimodal input token)",
        "action_groups": ACTION_GROUPS,
        "timestep": float(timestep),
        "num_projections": int(projections),
        "n_frames": len(records),
        "n_episodes": len(episodes),
        "n_subtasks": len(subtasks),
        "subtask_counts": counts,
        "camera_scales": {f"{group}|{camera}": value for (group, camera), value in camera_scales.items()},
        "comparison_rule": "compare the same action group and input segment across subtasks",
        "normalization": "none across modalities or action groups; flow-sample RMS only",
    }


def _run_dataset(adapter, dataset, cfg, output_dir) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if adapter.action_dim < 7:
        raise ValueError(
            f"The configured proximal/wrist/gripper grouping requires 7 actions; got {adapter.action_dim}."
        )
    samples = build_subtask_samples(dataset, cfg)
    if not samples:
        logging.warning("[action_sensitivity] no subtask-labelled frames were sampled.")
        return
    records = _capture_records(adapter, dataset, cfg, samples)
    camera_scales, prompt_scales = _global_scales(records)
    html_records = _write_details(output_dir, records)
    prompt_panels = _plot_prompt_maps(output_dir, records)
    _write_frame_metrics(output_dir, records)
    html_panel = _write_html(output_dir, html_records, camera_scales, prompt_scales)

    p = cfg.probe_parameters
    summary = _summary(
        records,
        camera_scales,
        int(getattr(p, "action_sensitivity_projections", 4)),
        float(getattr(p, "timestep", 0.5)),
    )
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    panels = [
        Panel(
            html_panel,
            "Detailed validation-frame sensitivity browser",
            how="A patch is one input token; its value is the Frobenius norm of d(flow "
                "velocity of the selected joint group)/d(that token). Pick a subtask and an "
                "action group; tick *Compare two frames* to put two anchors side by side on "
                "one shared ramp. Brightness is relative to the scale on each panel's "
                "colourbar — a dark patch is small compared to that scale, not unimportant — "
                "and hovering a patch reads out its raw value. Colour scale defaults to the "
                "run's p99 so a typical frame uses most of the ramp; *auto* spans only the "
                "frames on screen, *run max* is the unclipped scale. Contrast defaults to a "
                "gamma solved from the drawn values (median at mid-ramp) rather than a fixed "
                "exponent. Compare the same group and the same panel only; camera, prompt, "
                "and group scales are never unified.",
            primary=True,
        )
    ]
    for filename in prompt_panels:
        group = filename.removeprefix("prompt_").removesuffix(".png")
        panels.append(
            Panel(
                filename,
                f"{GROUP_LABELS.get(group, group)} prompt-token sensitivity",
                how="Bars are raw token Jacobian norms averaged within episode, then across episodes. Compare subtasks within this panel only.",
            )
        )
    panels.extend(
        [
            Panel("frame_metrics.csv", "Per-frame raw segment metrics",
                  how="One row per differentiated frame with its per-segment Jacobian "
                      "norms — the numbers the aggregate heatmaps average, for checking "
                      "whether a hot row is one frame or the whole subtask."),
            Panel("summary.json", "Probe definition, sampling counts, and scales",
                  how="Which frames were sampled, how many per subtask, and the fixed "
                      "colour scales the panels were drawn on. Read it before comparing "
                      "two runs' heatmaps by eye."),
        ]
    )
    write_index(
        output_dir,
        sys.modules[__name__],
        title="Subtask Action Sensitivity",
        claim="Do real image patches and prompt tokens influence arm, wrist, and gripper outputs differently across subtasks?",
        group="Sensitivity",
        status="info",
        summary=summary,
        metrics=[
            Metric("n_subtasks", "Subtasks compared", value=summary["n_subtasks"], fmt=0, primary=True,
                   good="none",
                   note="Distinct subtask clauses among the sampled frames. Two subtasks "
                        "cannot be compared on the aggregate maps unless both appear here."),
            Metric("n_episodes", "Episodes represented", value=summary["n_episodes"], fmt=0, primary=True,
                   good="none",
                   note="Aggregates are averaged within an episode before across episodes, so "
                        "a subtask carried by one episode is one sample however many frames it has."),
            Metric("n_frames", "Real frames differentiated", value=summary["n_frames"], fmt=0, primary=True,
                   good="none",
                   note="Validation frames the Jacobian was actually taken at — the sample size "
                        "behind every map here."),
            Metric("num_projections", "VJPs per action group", value=summary["num_projections"], fmt=0,
                   good="none",
                   note="Hutchinson probes averaged into each Frobenius-norm estimate. More is a "
                        "less noisy estimate at linear cost; it does not change what is measured."),
        ],
        panels=panels,
        see_also=["attention", "action_trace", "subtask_sweep"],
    )


def run(adapter, primary_dataset, cfg, output_dir):
    if adapter is None or primary_dataset is None:
        return
    try:
        _run_dataset(adapter, primary_dataset, cfg, output_dir)
    except NotImplementedError as exc:
        logging.warning("[action_sensitivity] %s — skipping.", exc)
        return
    for extra_root in getattr(cfg.dataset, "additional_offline_dataset_paths", []) or []:
        extra = load_extra_dataset(cfg.dataset.repo_id, extra_root)
        _run_dataset(
            adapter,
            extra,
            cfg,
            os.path.join(output_dir, os.path.basename(os.path.normpath(extra_root))),
        )


@parser.wrap()
def probe_cli(cfg: ProbeJacobianConfig):
    init_logging()
    output_dir = os.path.join(cfg.probe_parameters.output_dir, "action_drift_jacobian")
    normalization_dataset = load_probe_dataset(cfg)
    val_path = getattr(cfg, "val_dataset_path", None)
    dataset = (
        load_extra_dataset(cfg.dataset.repo_id, val_path)
        if val_path
        else normalization_dataset
    )
    if val_path:
        logging.info("[action_sensitivity] validation dataset: %s", val_path)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=normalization_dataset)
    run(adapter, dataset, cfg, output_dir)
    logging.info("Done. Output saved to %s", output_dir)


if __name__ == "__main__":
    register_config_choices()
    probe_cli()
