#!/usr/bin/env python
"""Representation trajectories: where do episodes approach other behaviors?

Capture temporal sequences with separate image, state, text and action token groups,
using the conditions-matrix adapter seam (first flow step, fixed noise). The default
view is the native attention input; normalization happens inside the model before
pooling. Real and constant task/subtask prompts are captured separately.

``trajectories.html`` follows one episode frame by frame: a slider over its sampled
frames drives both camera previews, a crosshair on the episode's full cosine matrix
(every frame against every other, in the original dimensions), the selected frame's
cosine profile along the episode, its best matches in other episodes and the aligned
cross-episode grid around one of those matches. Everything heavy is loaded per panel
from ``panels/<key>.js`` on demand; frame previews are read from ``cache/thumbs``.
Keep the directory together when moving the report.

Raw cosines between model states sit near 1 because all states share a common
direction, so three geometries are measured: raw, robot-centered (per-robot training
mean) and episode-centered (each episode's own sampled mean, leaving only motion
within the episode). To tell a trivially high similarity from structure, every frame
also records its *background*: the mean cosine to all frames of other episodes, same
and other robot. No PCA, UMAP, whitening, or similarity measured in a projection.

Matches exclude the entire source episode, and split by same / different robot and
task (exact task text). Each category keeps the best frame of each of the top
``MATCH_TOP_K`` episodes, so the report can browse alternatives. Target-window cosine
averages the anchor's similarity to frames near its match; the aligned grid in the
report is what shows whether a match persists in time. Crossings are candidates for
inspection, not a confusion score or causal evidence. These are dataset replays, not
closed-loop rollouts; text controls do not remove scene/state confounding. Native
action queries are pooled over future positions.

Sampling takes one frame every ``repr_stride_s`` seconds of every episode (the stored
image frame or corpus anchor nearest to each multiple of the stride), balances
episodes over robot, task and split, then keeps whole episodes in that order while
they fit under ``repr_max_frames``; per robot, held-out episodes come first under their own
cap ``repr_holdout_episodes`` (default all), training episodes under ``max_episodes``.
Episodes are never thinned below the stride, so
the aligned grid always has a frame near every offset it asks for. ``cache/`` saves
float32 selected layers and frame previews for model-free replotting; old
whole-sequence caches are deliberately rejected. ``sampling_plan.json``,
``temporal.csv`` and ``matches.csv`` retain the measurements.
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
    if not np.isfinite(p.repr_stride_s) or p.repr_stride_s <= 0:
        raise ValueError("repr_stride_s must be a positive number of seconds")
    if p.repr_min_frames_per_episode < 2 or p.repr_min_frames_per_episode > p.repr_max_frames:
        raise ValueError("repr_min_frames_per_episode must be >= 2 and <= repr_max_frames")
    return layers, groups


def stride_picks(times, stride_s):
    """Positions of the frame nearest to each multiple of the stride from the first frame.

    Targets run from the first time to the last in steps of ``stride_s``; each takes
    the nearest available frame, duplicates collapse, order is chronological. Nothing
    is interpolated: where the data is sparser than the stride, fewer frames come back.
    """
    t = np.asarray(times, dtype=np.float64)
    if len(t) == 0:
        return []
    order = np.argsort(t, kind="stable")
    sorted_t = t[order]
    targets = np.arange(sorted_t[0], sorted_t[-1] + stride_s / 2, stride_s)
    right = np.searchsorted(sorted_t, targets)
    left = np.clip(right - 1, 0, len(t) - 1)
    right = np.clip(right, 0, len(t) - 1)
    nearest = np.where(np.abs(sorted_t[right] - targets) < np.abs(sorted_t[left] - targets), right, left)
    return [int(order[k]) for k in dict.fromkeys(nearest.tolist())]


def _lerobot_candidates(dataset, cfg, key, holdout):
    """Read labels/times without decoding video; one frame per stride, no subtask-window restriction."""
    from lerobot.datasets.embodiment import embodiment_name

    identity = dataset_identity_columns(dataset, cfg)
    robot = embodiment_name(identity["embodiment_index"]) or f"embodiment {identity['embodiment_index']}"
    stride = probe_image_stride(cfg)
    columns = dataset.hf_dataset.select_columns(["frame_index", "timestamp", "task_index"]).with_format(None)[:]
    root = str(Path(dataset.root).resolve())
    candidates = []
    stride_s = cfg.probe_parameters.repr_stride_s
    for ep, indices in build_episode_index(dataset).items():
        frames = []
        available = [i for i in indices if i % stride == 0]
        for position in stride_picks([float(columns["timestamp"][i]) for i in available], stride_s):
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


def _diverse_candidates(buffer, key, holdout, stride_s):
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
        frames = []
        for position in stride_picks([float(row["anchor_s"]) for _, row in ordered], stride_s):
            index, row = ordered[position]
            frames.append({"kind": "diverse", "source_key": key, "source": row["source"],
                           "robot": str(row["embodiment"]), "episode": f"{corpus}::{ep}",
                           "episode_label": f"{row['source']}/{ep}", "index": index,
                           "frame": int(row["anchor_frame"]), "time_s": float(row["anchor_s"]), "task": task,
                           "holdout": holdout})
        if frames:
            candidates.append(frames)
    return candidates


def select_sequences(candidates, max_episodes, max_frames, seed, min_frames_per_episode=2,
                     max_holdout_episodes=None):
    """Balance robot/task/split, then keep whole episodes while they fit the frame budget.

    max_episodes is per robot and counts training episodes only; held-out episodes have
    their own per-robot cap ``max_holdout_episodes`` (None = all of them) and go first in
    each robot's queue. Source episode identity excludes duplicate copies and
    train/holdout overlap. Episodes shorter than ``min_frames_per_episode`` frames are
    left out; the others are taken in balanced round-robin order and an episode that
    no longer fits under ``max_frames`` is skipped, never thinned or cut short, so the
    frames of every kept episode stay at the sampling stride.
    """
    if max_frames < 2 or min_frames_per_episode < 2 or (max_episodes is not None and max_episodes < 1) \
            or (max_holdout_episodes is not None and max_holdout_episodes < 1):
        raise ValueError("Need >= 2 frames per episode and positive episode/frame budgets")
    rng = np.random.default_rng(seed)
    buckets = defaultdict(lambda: defaultdict(list))
    seen = set()
    for frames in candidates:
        if len(frames) < min_frames_per_episode:
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
        keys = sorted(cells)
        rng.shuffle(keys)
        queues[robot] = []
        for held, cap in ((True, max_holdout_episodes), (False, max_episodes)):
            split = [k for k in keys if k[0] == held]
            queue = []
            for i in range(max((len(cells[k]) for k in split), default=0)):
                for key in split:
                    if i < len(cells[key]):
                        queue.append(cells[key][i])
            queues[robot] += queue if cap is None else queue[:cap]
    chosen, used = [], 0
    for i in range(max((len(q) for q in queues.values()), default=0)):
        for queue in queues.values():
            if i < len(queue) and used + len(queue[i]) <= max_frames:
                chosen.append(queue[i])
                used += len(queue[i])
    if not chosen:
        raise ValueError(f"No episode with at least {min_frames_per_episode} frames fits under {max_frames} frames")
    samples = [frame for frames in chosen for frame in frames]
    return sorted(samples, key=lambda s: (s["kind"], s["source_key"], s["episode"], s["time_s"]))


def sampling_plan(dataset, cfg):
    p = cfg.probe_parameters
    datasets, buffers, candidates = {}, {}, []
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
            candidates += _diverse_candidates(buffer, key, key.endswith("holdout"), p.repr_stride_s)
    samples = select_sequences(candidates, p.max_episodes, p.repr_max_frames, p.random_seed,
                               p.repr_min_frames_per_episode, p.repr_holdout_episodes)
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
    meta = {"protocol": PROTOCOL, "view": p.repr_view, "layers": layers, "stride_s": p.repr_stride_s,
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
        plan = {"protocol": PROTOCOL, "stride_s": p.repr_stride_s, "samples": samples,
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
                               constant_text=p.repr_constant_text, cross_matrices=p.repr_cross_matrices)
    write_report(payload, root)
    write_index(str(root), sys.modules[__name__], title="Representation trajectories", group="Representation",
                claim="Where do episode trajectories approach other tasks in the original representation space?",
                summary=summary, status="info",
                metrics=[Metric("frames", "Frames", fmt=0), Metric("episodes", "Episodes", fmt=0),
                         Metric("robots", "Robots", fmt=0)],
                panels=[Panel("trajectories.html", "Frame slider over an episode: cameras, within-episode cosine matrix, frame profile against background, top matches in other episodes and the aligned cross-episode grid around a match",
                              how="Pick a geometry (robot- or episode-centered removes the shared direction that makes raw cosines ~1), then slide through the episode. Read the match lines against the background; pick a match and read the aligned grid: a bright diagonal through the center means the two episodes move through the same states at the same pace, a single bright cell means a momentary coincidence.", primary=True),
                        Panel("temporal.csv", "Original-space local cosine, background cosine, raw norms and sampling gaps"),
                        Panel("matches.csv", "Top cross-episode matches per category and neighborhood support"),
                        Panel("sampling_plan.json", "Exact sampled episodes and frames, and the sampling stride")],
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
