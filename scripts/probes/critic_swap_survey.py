#!/usr/bin/env python3
"""Run a bounded, source/task-stratified text-swap survey with one loaded critic.

All sources are loaded in full so root-relative annotation indices remain valid;
only the selected episodes are evaluated. The saved config pins the checkpoint.
"""
from __future__ import annotations

import copy
import gc
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs import parser
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.critic import ProbeCriticConfig, run_episode_critic_traces
from lerobot.probes.critic_sensitivity import _save_frame_images
from lerobot.probes.critic_subtasks import load_text_counts
from lerobot.probes.utils import build_episode_index, load_probe_dataset, register_config_choices
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class SurveyConfig(ProbeCriticConfig):
    survey_episodes: int = 128
    survey_prepare_only: bool = False
    survey_reuse_completed: bool = False


def select_episodes(sources, budget, seed):
    """Round-robin source/task groups, shuffled reproducibly within each group."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for source in sources:
        if float(source.weight) <= 0:
            continue
        root = Path(source.root)
        allowed = None if source.episodes is None else set(map(int, source.episodes))
        parts = sorted((root / "meta/episodes").rglob("*.parquet"))
        table = pd.concat([pd.read_parquet(p, columns=["episode_index", "tasks", "length"]) for p in parts], ignore_index=True)
        fps = float(json.loads((root / "meta/info.json").read_text())["fps"])
        for row in table.to_dict("records"):
            episode = int(row["episode_index"])
            if allowed is not None and episode not in allowed:
                continue
            tasks = [str(t) for t in row["tasks"]]
            groups[(source.name, tuple(tasks))].append({
                "source": source.name, "root": str(root), "episode": episode,
                "tasks": tasks, "length": int(row["length"]), "fps": fps,
            })
    keys = sorted(groups)
    rng.shuffle(keys)
    for rows in groups.values():
        rng.shuffle(rows)
    chosen = []
    for depth in range(max(map(len, groups.values()), default=0)):
        for key in keys:
            if depth < len(groups[key]):
                chosen.append(groups[key][depth])
                if len(chosen) == budget:
                    return chosen
    return chosen


def prepare(cfg, root, *, write=True):
    from lerobot.rl.offline_dataset_utils import get_offline_dataset_sources
    sources = get_offline_dataset_sources(cfg)
    plan = select_episodes(sources, int(cfg.survey_episodes), int(cfg.probe_parameters.random_seed))
    source_rows = defaultdict(list)
    for row in plan:
        source_rows[row["source"]].append(row)
    true_texts = set()
    for rows in source_rows.values():
        folder = Path(rows[0]["root"])
        windows = json.loads((folder / "meta/subtask_windows.json").read_text())["episodes"]
        for row in rows:
            true_texts.update(w["subtask"] for w in windows[str(row["episode"])])
    validation_root = Path(cfg.val_dataset_path)
    windows = json.loads((validation_root / "meta/subtask_windows.json").read_text())["episodes"]
    for rows in windows.values():
        true_texts.update(w["subtask"] for w in rows)
    step = int(Path(cfg.policy.pretrained_path).parent.name)
    counts = load_text_counts(cfg.probe_parameters.critic_text_counts_path)
    key = lambda s: " ".join(s.strip().split()).casefold().rstrip(".")
    normalized = {key(t) for t in true_texts}
    candidates = [t for t,n in counts.items() if n > 0 and key(t) not in normalized]
    if not candidates:
        raise ValueError("No shared off-label control is available.")
    control = min(candidates, key=lambda t: (-counts[t],t))
    provenance = {
        "checkpoint": str(cfg.policy.pretrained_path), "step": step,
        "sampling": "Round-robin source/task groups, seeded shuffle of episodes within each group",
        "seed": int(cfg.probe_parameters.random_seed), "split": "training",
        "stride_frames": int(cfg.probe_parameters.critic_trace_stride_frames),
        "control_text": control, "control_train_episodes": counts[control],
        "control_note": "Absent from selected annotated texts; semantic impossibility is not guaranteed.",
        "selected_episodes": plan,
        "training_episodes": len(plan), "source_task_groups": len({(r['source'],tuple(r['tasks'])) for r in plan}),
        "planned_train_points": sum((r["length"]+29)//30 for r in plan),
        "validation_root": str(validation_root), "status": "prepared",
    }
    root.mkdir(parents=True, exist_ok=True)
    if write:
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    return sources, source_rows, provenance


@parser.wrap()
def main(cfg: SurveyConfig):
    init_logging()
    root = Path(cfg.probe_parameters.output_dir)
    previous = json.loads((root / 'provenance.json').read_text()) if cfg.survey_reuse_completed else None
    sources, source_rows, provenance = prepare(cfg, root, write=not cfg.survey_reuse_completed)
    completed = []
    if previous is not None:
        for key in ('checkpoint', 'selected_episodes', 'control_text', 'stride_frames'):
            if previous[key] != provenance[key]:
                raise ValueError(f'Cannot reuse completed sources: {key} changed')
        completed = list(previous.get('completed_sources', []))
        for name in completed:
            for row in source_rows[name]:
                path = root / f"step_{provenance['step']:08d}/critic_text_swap/sources/{name}/episode_traces/ep{row['episode']:04d}/critic_values.json"
                saved = json.loads(path.read_text())['records']
                if len(saved) != (row['length'] + 29) // 30:
                    raise ValueError(f'Incomplete saved episode: {path}')
        provenance['completed_sources'] = completed
        provenance['reused_completed_sources'] = list(completed)
    provenance['camera_presence'] = 'Prepared per-episode camera roles; absent placeholders masked as in training'
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))

    logging.info("[SURVEY] %s training episodes, %s source/task groups, %s points, control: %s",
                 provenance['training_episodes'], provenance['source_task_groups'],
                 provenance['planned_train_points'], provenance['control_text'])
    for source, rows in source_rows.items():
        logging.info("[SURVEY] %s: %s episodes, %s frames", source, len(rows), sum(r['length'] for r in rows))
    if cfg.survey_prepare_only:
        return
    torch.cuda.set_per_process_memory_fraction(0.30)
    device = get_safe_torch_device(try_device=cfg.policy.device)
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg.dataset.root = cfg.val_dataset_path
    reference_cfg.dataset.episodes = None
    reference_cfg.dataset.max_episodes = None
    reference_cfg.probe_parameters.max_episodes = None
    reference_dataset = load_probe_dataset(reference_cfg)
    adapter = ProbablePolicy.for_config(cfg, device, dataset=reference_dataset)
    adapter.policy.model.to('cpu')
    if hasattr(adapter.policy, 'critic_target'):
        adapter.policy.critic_target.to('cpu')
    torch.cuda.empty_cache()
    probe_root = root / f"step_{provenance['step']:08d}/critic_text_swap"
    probe_root.mkdir(parents=True, exist_ok=True)
    provenance['status'] = 'running'
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    for source in sources:
        if source.name in completed:
            logging.info('[SURVEY] reusing completed source %s', source.name)
            continue
        rows = source_rows.get(source.name, [])
        if not rows:
            continue
        source_cfg = copy.deepcopy(cfg)
        source_cfg.val_dataset_path = None
        source_cfg.dataset.root = source.root
        source_cfg.dataset.episodes = None  # preserve root-relative annotation indexing
        source_cfg.dataset.max_episodes = None
        source_cfg.probe_parameters.max_episodes = None
        source_cfg.probe_parameters.critic_diverse_frames = 0
        dataset = load_probe_dataset(source_cfg)
        if int(dataset.fps) != 30:
            raise ValueError(f"Unexpected FPS for {source.name}: {dataset.fps}")
        index = build_episode_index(dataset)
        snapshots = {}
        for row in rows:
            points = index[row['episode']][::30]
            for q in (.2, .5, .8):
                snapshots[points[round(q*(len(points)-1))]] = f"{int(q*100)}% through episode"
        out = probe_root / 'sources' / source.name
        out.mkdir(parents=True, exist_ok=True)
        def observe(row, obs):
            row['source'] = source.name
            row['split'] = 'training'
            if row['global_idx'] in snapshots:
                row['snapshot'] = snapshots[row['global_idx']]
                images = _save_frame_images(obs, source_cfg, row, out)
                row['images'] = [{**im, 'path': str((out/im['path']).relative_to(probe_root))} for im in images]
        trace = run_episode_critic_traces(adapter, dataset, {r['episode'] for r in rows}, source_cfg,
                                         str(out/'episode_traces'), unrelated_text=provenance['control_text'],
                                         on_record=observe, include_fit=False)
        if trace is None:
            raise RuntimeError(f'No traces for {source.name}')
        completed.append(source.name)
        provenance['completed_sources'] = completed
        (root / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
        logging.info('[SURVEY] completed %s: %s points', source.name, len(trace['records']))
        del dataset, trace
        gc.collect()
    # Evaluate validation at the same checkpoint and with the same fixed control.
    ref_out = probe_root / 'sources/validation'
    ref_out.mkdir(parents=True, exist_ok=True)
    index = build_episode_index(reference_dataset)
    snapshots = {}
    for indices in index.values():
        points = indices[::30]
        for q in (.2, .5, .8):
            snapshots[points[round(q*(len(points)-1))]] = f"{int(q*100)}% through episode"
    def observe_validation(row, obs):
        row['source'] = 'validation'
        row['split'] = 'validation'
        if row['global_idx'] in snapshots:
            row['snapshot'] = snapshots[row['global_idx']]
            images = _save_frame_images(obs, reference_cfg, row, ref_out)
            row['images'] = [{**im, 'path': str((ref_out/im['path']).relative_to(probe_root))} for im in images]
    trace = run_episode_critic_traces(adapter, reference_dataset, None, reference_cfg,
                                     str(ref_out/'episode_traces'), unrelated_text=provenance['control_text'],
                                     on_record=observe_validation, include_fit=False)
    if trace is None:
        raise RuntimeError('No validation traces')
    provenance['validation_episodes'] = len(index)
    provenance['validation_points'] = len(trace['records'])
    provenance['status'] = 'evaluated'
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    from lerobot.probes.critic_text_swap import render
    summary = render(root)
    provenance['status'] = 'complete' if summary['complete'] else 'partial'
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
    logging.info('[SURVEY] evaluation complete: %s', probe_root)


if __name__ == '__main__':
    register_config_choices()
    main()
