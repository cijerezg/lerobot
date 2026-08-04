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

Outputs under ``action_drift_jacobian/``:

* ``sensitivity.html`` — detailed frame browser with camera overlays and prompt tokens.
* ``aggregate_<group>.png`` — episode-balanced spatial maps, one row per subtask.
* ``prompt_<group>.png`` — episode-balanced prompt-token sensitivity per subtask.
* ``frame_metrics.csv`` / ``summary.json`` — raw aggregate values and provenance.
* ``raw/*.npz`` — per-frame token scores for reproducible downstream analysis.
"""

import csv
import json
import logging
import math
import os
import random
import re
import sys
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
                grids[name] = grid["per_head_grid"].squeeze(0).numpy()
                grid_images[name] = grid
            group_data[group] = {
                "grids": grids,
                "grid_images": grid_images,
                "prompt": _prompt_entries(view),
                "scores": capture.scores_by_group[group].detach().float().cpu().numpy(),
            }
        records.append({**sample, "capture": capture, "groups": group_data})
    return records


def _global_scales(records) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    camera_scales: dict[tuple[str, str], float] = {}
    prompt_scales: dict[str, float] = dict.fromkeys(ACTION_GROUPS, 0.0)
    for record in records:
        for group, data in record["groups"].items():
            for camera, grid in data["grids"].items():
                key = (group, camera)
                camera_scales[key] = max(camera_scales.get(key, 0.0), float(np.max(grid)))
            for token in data["prompt"]:
                prompt_scales[group] = max(prompt_scales[group], float(token["value"]))
    camera_scales = {key: max(value, 1e-12) for key, value in camera_scales.items()}
    prompt_scales = {key: max(value, 1e-12) for key, value in prompt_scales.items()}
    return camera_scales, prompt_scales


def _write_details(output_dir, records, camera_scales) -> list[dict[str, Any]]:
    from PIL import Image

    from lerobot.probes.attention import _render_overlays_from_grids

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
        np.savez_compressed(
            os.path.join(output_dir, raw_rel),
            **{group: data["scores"] for group, data in record["groups"].items()},
        )

        groups_json: dict[str, Any] = {}
        for group, data in record["groups"].items():
            panels = []
            for camera, grid_info in data["grid_images"].items():
                vmax = camera_scales[(group, camera)]
                rendered, _ = _render_overlays_from_grids(
                    [grid_info],
                    vmax_overrides={
                        f"{camera}_mean": vmax,
                        f"{camera}_mean_vmin": 0.0,
                        f"{camera}_heads": [vmax],
                        f"{camera}_heads_vmin": [0.0],
                    },
                    mean_title_template="sensitivity: {camera}",
                )
                panel = rendered.get(f"overlay_{camera}_summary")
                if panel is None:
                    continue
                filename = f"{group}_{_slug(camera)}.jpg"
                absolute = os.path.join(frame_dir, filename)
                Image.fromarray(panel).save(absolute, quality=90)
                panels.append(
                    {
                        "camera": camera,
                        "file": os.path.relpath(absolute, output_dir),
                        "vmax": vmax,
                    }
                )
            groups_json[group] = {
                "panels": panels,
                "prompt": data["prompt"],
            }
        html_records.append(
            {
                "episode": record["episode"],
                "frame": record["frame"],
                "global_idx": record["global_idx"],
                "subtask": record["subtask"],
                "raw": raw_rel,
                "groups": groups_json,
            }
        )
    return html_records


def _episode_balanced_grid(records, subtask: str, group: str, camera: str):
    by_episode: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        if record["subtask"] != subtask:
            continue
        grid = record["groups"][group]["grids"].get(camera)
        if grid is not None:
            by_episode[int(record["episode"])].append(np.asarray(grid, dtype=np.float64))
    episode_maps = [np.mean(parts, axis=0) for parts in by_episode.values() if parts]
    if not episode_maps:
        return None, 0, 0
    return np.mean(episode_maps, axis=0), len(episode_maps), sum(len(v) for v in by_episode.values())


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


def _plot_aggregate_maps(output_dir, records, camera_scales) -> list[str]:
    subtasks = sorted({record["subtask"] for record in records})
    cameras = sorted(
        {camera for record in records for data in record["groups"].values() for camera in data["grids"]},
        key=lambda name: ("top" not in name, "wrist" not in name, name),
    )
    written = []
    for group in ACTION_GROUPS:
        if not cameras or not subtasks:
            continue
        fig, axes = plt.subplots(
            len(subtasks), len(cameras),
            figsize=(3.2 * len(cameras), 2.6 * len(subtasks)),
            squeeze=False,
        )
        for row, subtask in enumerate(subtasks):
            for col, camera in enumerate(cameras):
                ax = axes[row, col]
                mean_grid, n_episodes, n_frames = _episode_balanced_grid(
                    records, subtask, group, camera
                )
                if mean_grid is None:
                    ax.axis("off")
                    continue
                side = int(round(math.sqrt(mean_grid.size)))
                display = mean_grid.reshape(side, side) if side * side == mean_grid.size else mean_grid
                image = ax.imshow(
                    display,
                    cmap="magma",
                    vmin=0.0,
                    vmax=camera_scales[(group, camera)],
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    f"{camera} · {n_episodes} eps / {n_frames} frames\n"
                    f"raw max scale {camera_scales[(group, camera)]:.2e}",
                    fontsize=8,
                )
                if col == 0:
                    ax.set_ylabel(subtask, fontsize=8)
                fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        fig.suptitle(
            f"{GROUP_LABELS[group]} sensitivity by subtask\n"
            "episode-balanced means; each camera has one fixed raw scale across all subtasks",
            fontsize=11,
        )
        fig.tight_layout()
        filename = f"aggregate_{group}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=145, bbox_inches="tight")
        plt.close(fig)
        written.append(filename)
    return written


def _plot_prompt_maps(output_dir, records) -> list[str]:
    subtasks = sorted({record["subtask"] for record in records})
    written = []
    for group in ACTION_GROUPS:
        fig, axes = plt.subplots(
            len(subtasks), 1,
            figsize=(9.0, max(2.5, 2.6 * len(subtasks))),
            squeeze=False,
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
        fig.suptitle(f"{GROUP_LABELS[group]} sensitivity to prompt tokens", fontsize=11)
        fig.tight_layout()
        filename = f"prompt_{group}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=145, bbox_inches="tight")
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


def _write_html(output_dir, html_records, camera_scales, prompt_scales) -> str:
    payload = {
        "records": html_records,
        "groups": list(ACTION_GROUPS),
        "groupLabels": GROUP_LABELS,
        "cameraScales": {f"{group}|{camera}": value for (group, camera), value in camera_scales.items()},
        "promptScales": prompt_scales,
    }
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Subtask action sensitivity</title>
<style>
:root{{--bg:#f5f5f1;--card:#fff;--ink:#18181b;--muted:#71717a;--line:#deded8;--accent:#6d28d9}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}}
header{{padding:22px 28px 14px;border-bottom:1px solid var(--line);background:#fafaf8}}
h1{{font-size:22px;margin:0 0 5px}} .dek{{color:var(--muted);max-width:1000px}}
.controls{{display:grid;grid-template-columns:minmax(220px,2fr) minmax(150px,1fr) minmax(140px,1fr) 2fr;gap:12px;padding:14px 28px;position:sticky;top:0;background:rgba(245,245,241,.96);backdrop-filter:blur(8px);z-index:5;border-bottom:1px solid var(--line)}}
label{{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}} select,input{{width:100%;margin-top:4px;padding:8px;border:1px solid var(--line);border-radius:7px;background:white}}
main{{padding:20px 28px 40px;max-width:1600px;margin:auto}} .meta{{display:flex;gap:18px;align-items:baseline;margin-bottom:15px}} .subtask{{font-size:18px;font-weight:700}}
.camera-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:14px}} .panel{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;box-shadow:0 1px 2px #00000008}}
.panel img{{width:100%;display:block;border-radius:5px}} .caption{{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-top:7px}}
.prompt{{margin-top:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}} .tokens{{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}} .token{{border:1px solid #ddd6fe;border-radius:5px;padding:4px 6px;background:#faf5ff}} .token small{{display:block;color:#6b21a8;font-size:9px}}
.empty{{color:var(--muted);padding:30px;text-align:center}} .raw{{margin-left:auto}}
@media(max-width:850px){{.controls{{grid-template-columns:1fr 1fr}}.camera-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Subtask-conditioned action sensitivity</h1>
<div class="dek">Raw local Jacobian norms at real validation frames. Compare the same action group and input panel across subtasks; camera, prompt, and action-group scales are intentionally not unified.</div></header>
<div class="controls">
  <div><label>Subtask<select id="subtask"></select></label></div>
  <div><label>Action group<select id="group"></select></label></div>
  <div><label>Episode<select id="episode"></select></label></div>
  <div><label>Frame <span id="frameCount"></span><input id="frame" type="range" min="0" value="0"></label></div>
</div>
<main><div class="meta"><span class="subtask" id="title"></span><span id="where"></span><a class="raw" id="raw">raw NPZ</a></div>
<div class="camera-grid" id="cameras"></div>
<section class="prompt"><strong>Prompt-token sensitivity</strong><div class="tokens" id="tokens"></div></section></main>
<script>const DATA={data_json};
const $=id=>document.getElementById(id), sub=$('subtask'), group=$('group'), episode=$('episode'), slider=$('frame');
const subtasks=[...new Set(DATA.records.map(r=>r.subtask))].sort();
sub.innerHTML=subtasks.map(s=>`<option>${{s}}</option>`).join('');
group.innerHTML=DATA.groups.map(g=>`<option value="${{g}}">${{DATA.groupLabels[g]}}</option>`).join('');
function episodes(){{const vals=[...new Set(DATA.records.filter(r=>r.subtask===sub.value).map(r=>r.episode))].sort((a,b)=>a-b);episode.innerHTML='<option value="all">all episodes</option>'+vals.map(e=>`<option>${{e}}</option>`).join('')}}
function filtered(){{return DATA.records.filter(r=>r.subtask===sub.value&&(episode.value==='all'||String(r.episode)===episode.value))}}
function color(value,max){{const x=Math.max(0,Math.min(1,value/max));return `rgba(124,58,237,${{0.08+0.72*x}})`}}
function render(){{const rows=filtered();slider.max=Math.max(0,rows.length-1);slider.value=Math.min(+slider.value,+slider.max);$('frameCount').textContent=`${{rows.length}} sampled`;
 if(!rows.length){{$('cameras').innerHTML='<div class="empty">No sampled frames.</div>';return}} const r=rows[+slider.value],g=group.value,d=r.groups[g];
 $('title').textContent=r.subtask;$('where').textContent=`episode ${{r.episode}} · frame ${{r.frame}} · global ${{r.global_idx}}`;$('raw').href=r.raw;
 $('cameras').innerHTML=d.panels.map(p=>`<article class="panel"><img src="${{p.file}}"><div class="caption"><b>${{p.camera}}</b><span>fixed raw max ${{p.vmax.toExponential(2)}}</span></div></article>`).join('')||'<div class="empty">No camera patches mapped.</div>';
 const max=DATA.promptScales[g];$('tokens').innerHTML=d.prompt.map(t=>`<span class="token" style="background:${{color(t.value,max)}}" title="raw ${{t.value.toExponential(4)}}"><small>${{t.clause}}</small>${{t.label}}</span>`).join('')||'<span class="empty">No prompt tokens mapped.</span>';}}
sub.onchange=()=>{{episodes();slider.value=0;render()}};episode.onchange=()=>{{slider.value=0;render()}};group.onchange=render;slider.oninput=render;episodes();render();
</script></body></html>"""
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
    html_records = _write_details(output_dir, records, camera_scales)
    aggregate_panels = _plot_aggregate_maps(output_dir, records, camera_scales)
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
            how="Choose a subtask and one of the three action groups. Every camera uses one fixed raw scale across subtasks; prompt has its own scale.",
            primary=True,
        )
    ]
    for filename in aggregate_panels:
        group = filename.removeprefix("aggregate_").removesuffix(".png")
        panels.append(
            Panel(
                filename,
                f"{GROUP_LABELS.get(group, group)} spatial sensitivity by subtask",
                how="Rows are subtasks. A cell is the episode-balanced mean of raw patch Jacobian norms; compare vertically within one camera column.",
                primary=True,
            )
        )
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
            Metric("n_subtasks", "Subtasks compared", value=summary["n_subtasks"], fmt=0, primary=True),
            Metric("n_episodes", "Episodes represented", value=summary["n_episodes"], fmt=0, primary=True),
            Metric("n_frames", "Real frames differentiated", value=summary["n_frames"], fmt=0, primary=True),
            Metric("num_projections", "VJPs per action group", value=summary["num_projections"], fmt=0),
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
    probe_cli()
