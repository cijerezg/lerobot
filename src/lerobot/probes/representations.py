#!/usr/bin/env python
"""Representation trajectories: where do episodes approach other behaviors?

Capture temporal sequences with separate image, state, text and action token groups,
using the conditions-matrix adapter seam (first flow step, fixed noise). The default
view is the native attention input; normalization happens inside the model before
pooling. Real and constant task/subtask prompts are captured separately.

``trajectories.html`` follows one episode frame by frame: a slider over its sampled
frames drives both camera previews, a crosshair on the episode's full cosine matrix
(every frame against every other, in the original dimensions), the selected frame's
cosine profile along the episode, and a highlight in a shared 3-D PCA. Everything
heavy is loaded per panel from ``panels/<key>.js`` on demand; frame previews are
read from ``cache/thumbs``. Keep the directory together when moving the report.

Raw cosines between model states sit near 1 because all states share a common
direction, so three geometries are measured: raw, robot-centered (per-robot training
mean) and episode-centered (each episode's own sampled mean, leaving only motion
within the episode). To tell a trivially high local cosine from structure, every
frame also records its *background*: the mean cosine to all frames of other
episodes, same and other robot, and its best cross-episode matches. The local
radius is a slider in the report, recomputed from the matrix; the ``lags.csv``
profile gives mean cosine by time gap per episode, which says at what separation
frames stop being closer than other episodes are. PCA is fitted once per
group/layer/text/geometry across robots on training rows, then held fixed while
filtering; a robot with only held-out rows uses those rows descriptively, recorded
in the report. No UMAP, whitening, or similarity measured in the projection. Bases
are refitted across checkpoints.

Matches exclude the entire source episode, and split by same / different robot and
task (exact task text). Target-window cosine averages the anchor's similarity to
frames near its best match; this is not trajectory alignment. Crossings are
candidates for inspection, not a confusion score or causal evidence. These are
dataset replays, not closed-loop rollouts; text controls do not remove scene/state
confounding. Native action queries are pooled over future positions.

Sampling balances episodes over robot, task and split before distributing the frame
budget. LeRobot rows follow the stored image grid; diverse rows use available corpus
anchors. Sparse neighborhoods remain missing, with time gaps and counts recorded.
``cache/`` saves float32 selected layers and frame previews for model-free replotting;
old whole-sequence caches are deliberately rejected. ``sampling_plan.json``,
``temporal.csv``, ``matches.csv``, ``lags.csv`` and ``pca/`` retain the measurements.
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.conditions_matrix import (
    GROUPS, NEUTRAL_SUBTASK, NEUTRAL_TASK, _open_diverse_buffers, _rebot_inputs, _save_thumbs,
)
from lerobot.probes.domain_representations import _diverse_inputs
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.representation_views import VIEW_METADATA
from lerobot.probes.utils import (
    build_episode_index, dataset_identity_columns, load_extra_dataset, load_probe_dataset,
    probe_image_stride, register_config_choices,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

PROTOCOL = "temporal_token_groups_v1"


@dataclass
class ProbeRepresentationsConfig(TrainRLServerPipelineConfig):
    """Options live under probe_parameters."""


def _options(cfg):
    p = cfg.probe_parameters
    layers = list(dict.fromkeys(int(s.strip()) for s in p.repr_layers.split(",") if s.strip()))
    groups = list(dict.fromkeys(s.strip() for s in p.repr_groups.split(",") if s.strip()))
    if not layers or min(layers) < 0:
        raise ValueError("repr_layers must contain nonnegative block indices")
    if not groups or set(groups) - {g for _, g in GROUPS}:
        raise ValueError(f"repr_groups must be selected from {[g for _, g in GROUPS]}")
    if p.repr_view not in VIEW_METADATA:
        raise ValueError(f"Unknown repr_view {p.repr_view!r}")
    if not np.isfinite(p.repr_local_window_s) or p.repr_local_window_s <= 0 or p.repr_max_frames < 2:
        raise ValueError("repr_local_window_s must be positive; repr_max_frames must be >= 2")
    return layers, groups


def _lerobot_candidates(dataset, cfg, key, holdout):
    """Read labels/times without decoding video; no subtask-window restriction."""
    from lerobot.datasets.embodiment import embodiment_name

    identity = dataset_identity_columns(dataset, cfg)
    robot = embodiment_name(identity["embodiment_index"]) or f"embodiment {identity['embodiment_index']}"
    stride = probe_image_stride(cfg)
    columns = dataset.hf_dataset.select_columns(["frame_index", "timestamp", "task_index"]).with_format(None)[:]
    root = str(Path(dataset.root).resolve())
    candidates = []
    p = cfg.probe_parameters
    budget = p.repr_n_frames_per_episode if p.repr_n_frames_per_episode is not None else p.n_frames_per_episode
    for ep, indices in build_episode_index(dataset).items():
        frames = []
        available = [i for i in indices if i % stride == 0]
        picks = np.linspace(0, len(available) - 1, min(len(available), budget), dtype=int)
        for position in picks:
            index = available[position]
            task = str(dataset.meta.tasks.iloc[int(columns["task_index"][index])].name)
            frames.append({**identity, "kind": "lerobot", "source_key": key,
                           "source": Path(root).name, "robot": robot,
                           "episode": f"{root}::{ep}", "episode_label": f"{Path(root).name}/{ep}",
                           "index": index, "frame": int(columns["frame_index"][index]),
                           "time_s": float(columns["timestamp"][index]), "task": task,
                           "holdout": holdout})
        if frames:
            candidates.append(frames)
    return candidates


def _diverse_candidates(buffer, key, holdout, frames_per_episode):
    from lerobot.datasets.diverse_prompt import episode_task

    by_episode = defaultdict(list)
    for index, row in enumerate(buffer.rows):
        by_episode[(row["corpus_key"], str(row["episode_id"]))].append((index, row))
    candidates = []
    for (corpus, ep), items in sorted(by_episode.items()):
        first = items[0][1]
        task = episode_task(buffer.selection.episode_records[ep], first["source"])[0]
        ordered = sorted(items, key=lambda item: float(item[1]["anchor_s"]))
        ordered = list({float(row["anchor_s"]): (index, row) for index, row in ordered}.values())
        picks = np.linspace(0, len(ordered) - 1, min(len(ordered), frames_per_episode), dtype=int)
        frames, seen = [], set()
        for position in picks:
            index, row = ordered[position]
            t = float(row["anchor_s"])
            if t in seen:
                continue
            seen.add(t)
            frames.append({"kind": "diverse", "source_key": key, "source": row["source"],
                           "robot": str(row["embodiment"]), "episode": f"{corpus}::{ep}",
                           "episode_label": f"{row['source']}/{ep}", "index": index,
                           "frame": int(row["anchor_frame"]), "time_s": t, "task": task,
                           "holdout": holdout})
        if frames:
            candidates.append(frames)
    return candidates


def select_sequences(candidates, max_episodes, frames_per_episode, max_frames, seed):
    """Balance robot/task/split, then thin whole trajectories including both ends.

    max_episodes is per robot. Source episode identity excludes duplicate copies and
    train/holdout overlap. A hard budget never truncates just an episode's tail.
    """
    if frames_per_episode < 2 or max_frames < 2 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("Need >= 2 frames per episode and positive episode/frame budgets")
    rng = np.random.default_rng(seed)
    buckets = defaultdict(lambda: defaultdict(list))
    seen = set()
    for frames in candidates:
        if len(frames) < 2:
            continue
        frames = sorted(frames, key=lambda r: r["time_s"])
        first = frames[0]
        if first["episode"] in seen:
            raise ValueError(f"Episode appears in multiple sources/splits: {first['episode']}")
        seen.add(first["episode"])
        task_key = tuple(sorted({r["task"] for r in frames}))
        buckets[first["robot"]][(first["holdout"], task_key)].append(frames)
    queues = {}
    for robot, cells in sorted(buckets.items()):
        for cell in cells.values():
            rng.shuffle(cell)
        queue, keys = [], sorted(cells)
        rng.shuffle(keys)
        for i in range(max(map(len, cells.values()))):
            for key in keys:
                if i < len(cells[key]):
                    queue.append(cells[key][i])
        queues[robot] = queue if max_episodes is None else queue[:max_episodes]
    chosen = []
    for i in range(max((len(q) for q in queues.values()), default=0)):
        for queue in queues.values():
            if i < len(queue) and len(chosen) < max_frames // 2:
                chosen.append(queue[i])
    if not chosen:
        raise ValueError("No episodes with at least two available frames")
    counts = np.full(len(chosen), 2)
    limits = np.array([min(len(ep), frames_per_episode) for ep in chosen])
    remaining = max_frames - int(counts.sum())
    while remaining and np.any(counts < limits):
        for i in np.flatnonzero(counts < limits):
            if not remaining:
                break
            counts[i] += 1
            remaining -= 1
    samples = []
    for frames, count in zip(chosen, counts, strict=True):
        picks = np.linspace(0, len(frames) - 1, int(count), dtype=int)
        samples.extend(frames[i] for i in picks)
    return sorted(samples, key=lambda s: (s["kind"], s["source_key"], s["episode"], s["time_s"]))


def sampling_plan(dataset, cfg):
    p = cfg.probe_parameters
    datasets, buffers, candidates = {}, {}, []
    n_frames = p.repr_n_frames_per_episode if p.repr_n_frames_per_episode is not None else p.n_frames_per_episode
    if n_frames < 2:
        raise ValueError("Need >= 2 representation frames per episode")
    roots = [str(s.root) for s in (getattr(cfg.dataset, "sources", None) or []) if s.root]
    roots += list(getattr(cfg.dataset, "additional_offline_dataset_paths", None) or [])
    if getattr(cfg.dataset, "root", None):
        roots.append(str(cfg.dataset.root))
    if dataset is not None:
        roots.append(str(dataset.root))
    val_root = getattr(cfg, "val_dataset_path", None)
    if val_root:
        roots.append(str(val_root))
    for root in dict.fromkeys(str(Path(r).resolve()) for r in roots):
        ds = dataset if dataset is not None and str(Path(dataset.root).resolve()) == root else load_extra_dataset(cfg.dataset.repo_id, root)
        datasets[root] = ds
        held = bool(val_root and str(Path(val_root).resolve()) == root)
        candidates += _lerobot_candidates(ds, cfg, root, held)
    if getattr(cfg, "diverse", None) is not None and cfg.diverse.enabled:
        buffers = _open_diverse_buffers(cfg)
        for key, buffer in buffers.items():
            candidates += _diverse_candidates(buffer, key, key.endswith("holdout"), n_frames)
    samples = select_sequences(candidates, p.max_episodes, n_frames, p.repr_max_frames, p.random_seed)
    return samples, datasets, buffers


@torch.no_grad()
def collect(adapter, cfg, samples, datasets, buffers, cache_dir):
    layers, groups = _options(cfg)
    p = cfg.probe_parameters
    cache_dir = Path(cache_dir)
    (cache_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").unlink(missing_ok=True)
    texts = ("real", "neutral") if p.repr_constant_text else ("real",)
    arrays, rows = {}, []
    present = {g: np.zeros(len(samples) * len(texts), dtype=bool) for g in groups}
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            logging.info("Representations: %s/%s frames", i + 1, len(samples))
        inputs = (_rebot_inputs(datasets[sample["source_key"]], cfg, sample, adapter.chunk_size)
                  if sample["kind"] == "lerobot" else _diverse_inputs(buffers[sample["source_key"]], cfg, sample))
        # A camera absent in a new capture must not inherit a previous run's preview.
        for camera in ("external_0", "wrist_0"):
            (cache_dir / "thumbs" / f"{i:04d}.{camera}.jpg").unlink(missing_ok=True)
        _save_thumbs(inputs["obs"], str(cache_dir), i)
        for text in texts:
            real = text == "real"
            reps = adapter.capture_layer_representations(
                inputs["obs"], inputs["task"] if real else NEUTRAL_TASK,
                subtask=inputs["subtask"] if real else NEUTRAL_SUBTASK,
                metadata=inputs["metadata"], extra_complementary=inputs["extra"],
                noise_seed=int(p.random_seed), output_dtype=torch.float32,
                capture_native_views=p.repr_view != "block_output",
            )
            if p.repr_view != "block_output":
                reps = reps.get("native_views", {}).get(p.repr_view)
                if reps is None:
                    raise ValueError(f"Adapter did not capture native view {p.repr_view}")
            r = len(rows)
            for site, group in GROUPS:
                if group not in groups:
                    continue
                vec = reps.get(site, {}).get(group)
                if vec is None:
                    continue
                if max(layers) >= vec.shape[0]:
                    raise ValueError(f"repr_layers={layers} exceeds {group}'s {vec.shape[0]} blocks")
                values = vec[layers].detach().float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite representation at sample {i}/{group}")
                if group not in arrays:
                    arrays[group] = np.lib.format.open_memmap(
                        cache_dir / f"{group}.npy", mode="w+", dtype=np.float32,
                        shape=(len(samples) * len(texts), *values.shape))
                arrays[group][r] = values
                present[group][r] = True
            rows.append({**sample, "row": r, "sample": i, "text": text,
                         "task": inputs["task"], "subtask": inputs["subtask"]})
    if not arrays:
        raise ValueError("No requested token groups were captured")
    for array in arrays.values():
        array.flush()
    for group, mask in present.items():
        np.save(cache_dir / f"{group}.present.npy", mask)
    meta = {"protocol": PROTOCOL, "view": p.repr_view, "layers": layers,
            "groups": list(arrays), "requested_groups": groups, "rows": rows, "noise_seed": p.random_seed,
            "checkpoint": str(getattr(cfg.policy, "pretrained_path", "")), "flow_time": 0}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_cache(cache_dir):
    root = Path(cache_dir)
    if not (root / "meta.json").exists():
        raise ValueError("No temporal representation cache. Collect again; old whole-sequence caches are incompatible.")
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible representation cache protocol; collect again")
    arrays, present = {}, {}
    for group in meta["groups"]:
        arrays[group] = np.load(root / f"{group}.npy", mmap_mode="r")
        present[group] = np.load(root / f"{group}.present.npy")
        if arrays[group].shape[:2] != (len(meta["rows"]), len(meta["layers"])) or present[group].shape != (len(meta["rows"]),):
            raise ValueError(f"Incomplete cache for {group}; collect again")
    return meta, arrays, present


def run(adapter, primary_dataset, cfg, output_dir):
    p = cfg.probe_parameters
    layers, groups = _options(cfg)
    if p.mode not in ("collect", "plot", "all"):
        raise ValueError(f"Unknown probe mode {p.mode}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if p.mode in ("collect", "all"):
        if adapter is None:
            raise ValueError("Representation collection requires a policy adapter")
        samples, datasets, buffers = sampling_plan(primary_dataset, cfg)
        plan = {"protocol": PROTOCOL, "samples": samples,
                "episodes": len({r['episode'] for r in samples}), "frames": len(samples)}
        (root / "sampling_plan.json").write_text(json.dumps(plan, indent=2))
        collect(adapter, cfg, samples, datasets, buffers, root / "cache")
    if p.mode == "collect":
        return None
    from lerobot.probes.representation_report import write_report
    from lerobot.probes.representation_trajectories import analyze

    meta, arrays, present = load_cache(root / "cache")
    if (p.repr_view != meta["view"] or set(layers) - set(meta["layers"])
            or set(groups) - set(meta["requested_groups"])):
        raise ValueError("Requested view/layers/groups differ from cache; collect again or select cached values")
    if p.repr_constant_text and not any(r["text"] == "neutral" for r in meta["rows"]):
        raise ValueError("Cache has no constant-text control; collect again or set repr_constant_text=false")
    payload, summary = analyze(meta, arrays, present, layers, groups, p.repr_local_window_s, root,
                               constant_text=p.repr_constant_text)
    write_report(payload, root)
    write_index(str(root), sys.modules[__name__], title="Representation trajectories", group="Representation",
                claim="Where do episode trajectories approach other tasks in the original representation space?",
                summary=summary, status="info",
                metrics=[Metric("frames", "Frames", fmt=0), Metric("episodes", "Episodes", fmt=0),
                         Metric("robots", "Robots", fmt=0)],
                panels=[Panel("trajectories.html", "Frame slider over an episode: cameras, within-episode cosine matrix, frame profile against background, shared 3-D PCA and cross-episode matches",
                              how="Pick a geometry (robot- or episode-centered removes the shared direction that makes raw cosines ~1), then slide through the episode. Read the profile against the background and best-match lines. PCA crossings alone do not establish confusion.", primary=True),
                        Panel("temporal.csv", "Original-space local cosine, background cosine, raw norms and sampling gaps"),
                        Panel("matches.csv", "Cross-episode matches and neighborhood support"),
                        Panel("lags.csv", "Mean within-episode cosine by time gap between frames, per episode"),
                        Panel("sampling_plan.json", "Exact sampled episodes and frames")],
                see_also=["conditions_matrix", "subspace_spans", "action_trace"])
    return summary


@parser.wrap()
def probe_cli(cfg: ProbeRepresentationsConfig):
    init_logging()
    adapter = dataset = None
    if cfg.probe_parameters.mode in ("collect", "all"):
        dataset = load_probe_dataset(cfg)
        device = get_safe_torch_device(try_device=cfg.policy.device)
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    step = next((int(part) for part in reversed(Path(str(cfg.policy.pretrained_path or "")).parts) if part.isdigit()), 0)
    run(adapter, dataset, cfg, Path(cfg.probe_parameters.output_dir) / "validation" / f"step_{step:08d}" / "representations")


def main():
    register_config_choices()
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml

    config_path, remaining = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining, f"--config_path={_preprocess_config_yaml(config_path)}"]
    probe_cli()


if __name__ == "__main__":
    main()
