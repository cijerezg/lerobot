#!/usr/bin/env python
"""Set representations: one checkpoint's hidden state on frames from several datasets,
side by side in one cosine matrix with the frames on its axes.

**Question.** Does the policy represent our own robot's episodes, the external ReBot
recordings and the diverse-corpus robots in one region of the hidden state or in
separate ones, and in which token group and at what depth do the sets merge or split?

**Sets.** Three named sets, every frame through the same deployment prompt (subtask
clause from the annotation, deployment metadata clause) and one forward at flow time 0
(the action tokens are the same fixed noise for every frame), captured at
``setrep_view`` of the blocks in ``setrep_layers`` and pooled per token group:

* ``rebot``: ``setrep_rebot_episodes`` held-out episodes of the val root, distinct tasks
  where the root has them, ``setrep_rebot_frames`` frames evenly spaced over each;
* ``external``: ``setrep_external_episodes`` episodes of the external ReBot source (the
  config source whose name starts with ``external``), ``setrep_external_frames`` each;
* ``diverse``: one episode per entry of ``setrep_diverse_sources`` (``source:embodiment``
  pairs of the diverse corpus, chosen for different robots and layouts; a third field
  ``:n`` asks for n distinct episodes of that robot, ``:ep=id|id`` for those episodes),
  ``setrep_diverse_frames`` anchors evenly spaced over it, through the training diverse
  buffer exactly as training reads it; ``setrep_diverse_holdout`` (same syntax) adds
  episodes of the corpus holdout (never trained on, decoded from video), every row
  tagged ``split`` = train / holdout;
* ``named`` (optional): clips listed in ``setrep_named`` as ``label=root:ep:t0-t1@stride``,
  every frame of the window at that stride, for recurrence matrices and chosen reference
  frames (a hero rollout, a wrist-roll window). A raw ``inference_dataset`` root has no
  annotation; its subtask is the one the policy was given, from ``meta/online_labels.parquet``.

``setrep_rebot_sets`` / ``setrep_external_sets`` (``root:ep,ep;root:ep``) replace the random
picks with explicit episodes from any annotated root. ``setrep_balance_phases`` splits each episode's
frames evenly over the subtask verbs present (grasp / move / release / other) instead of
evenly over time, so a set is not two thirds grasp frames.

Episodes are the unit. Every within-set and between-set number below is a mean over
pairs of frames from DIFFERENT episodes, so a set's own diagonal says how alike two
episodes of that set are, not how alike a frame is to its temporal neighbours.

**Readouts** per token group and layer, in two geometries, raw and centered on the
mean of all sampled frames (the shared direction that puts every raw cosine near 1 is
removed once, over every set together; per-set or per-robot centering would subtract
the very difference being measured), and two measures, cosine (direction) and L2
distance (direction and magnitude, from the stored cosine and the two frames' norms in
the chosen geometry):

* the full frame-by-frame cosine matrix, rows ordered set, episode, time, thin lines
  between episodes and thick lines between sets, both frames of any cell on hover;
* the episode block matrix: mean cosine between every pair of episodes;
* the set block matrix: mean cosine between every pair of sets over cross-episode pairs;
* the gap per set pair: for cosine the two sets' within-set means averaged minus their
  between-set mean, for L2 the between-set mean minus the within-set average, so a
  positive gap always means "more alike within than between". A gap near 0 means the
  policy does not tell the sets apart in that group at that depth; a cosine gap near the
  within-set value means they never meet.

The headline numbers are the centered gaps at the deepest selected layer for the
external-camera image patches and for the action-expert tokens.

**Caveats.** Prompts differ by set by construction (task and subtask text, state width,
camera count, depth presence), so ``subtask`` and ``state`` separate the sets partly for
that reason; the image groups and ``action`` are where a shared or split representation
is informative. One episode per diverse source is one layout; its numbers say where
that layout sits, not the corpus. Dataset replays, not closed-loop rollouts.

Standalone::

    uv run python -m lerobot.probes.set_representations --config_path=config_rl_validate.yaml \\
        --policy.pretrained_path <ckpt>/pretrained_model --probe_parameters.output_dir <out>

``mode: plot`` re-reads ``cache/`` with no model.
"""
from __future__ import annotations

import csv
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
from lerobot.probes.conditions_matrix import GROUPS, _open_diverse_buffers, _rebot_inputs, _save_thumbs
from lerobot.probes.domain_representations import _diverse_inputs
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.representation_views import VIEW_METADATA
from lerobot.probes.utils import (
    build_episode_index, dataset_identity_columns, load_extra_dataset, probe_image_stride,
    register_config_choices,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging

PROTOCOL = "set_representations_v1"
SETS = ("rebot", "external", "diverse", "named")
GEOMETRIES = ("raw", "centered")
MEASURES = ("cosine", "distance")
HEADLINE_GROUPS = ("img_external_0", "action")


@dataclass
class ProbeSetRepresentationsConfig(TrainRLServerPipelineConfig):
    """Options live under probe_parameters (setrep_*)."""


def _options(cfg):
    p = cfg.probe_parameters
    layers = list(dict.fromkeys(int(s.strip()) for s in p.setrep_layers.split(",") if s.strip()))
    groups = list(dict.fromkeys(s.strip() for s in p.setrep_groups.split(",") if s.strip()))
    if not layers or min(layers) < 0:
        raise ValueError("setrep_layers must contain nonnegative block indices")
    if not groups or set(groups) - {g for _, g in GROUPS}:
        raise ValueError(f"setrep_groups must be selected from {[g for _, g in GROUPS]}")
    if p.setrep_view not in VIEW_METADATA:
        raise ValueError(f"Unknown setrep_view {p.setrep_view!r}")
    return layers, groups


def parse_sources(text):
    """``"droid_success:Franka,robochallenge:ARX5:2,yam:YAM:ep=yam__duster__ep000106|yam__duster__ep000174"``
    -> [("droid_success", "Franka", 1), ("robochallenge", "ARX5", 2), ("yam", "YAM", ["yam__duster__ep000106",
    "yam__duster__ep000174"])]: the third field is a count of random episodes (default 1) or ``ep=`` with
    explicit episode ids joined by ``|``."""
    pairs = []
    for item in text.split(","):
        if not item.strip():
            continue
        parts = [x.strip() for x in item.strip().split(":")]
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"setrep_diverse_sources entries are source:embodiment[:episodes], got {item!r}")
        episodes = parts[2] if len(parts) > 2 and parts[2] else "1"
        pairs.append((parts[0], parts[1], episodes[3:].split("|") if episodes.startswith("ep=") else int(episodes)))
    return pairs


def parse_rebot_sets(text):
    """``"root_a:0,3;root_b:4"`` -> [("root_a", [0, 3]), ("root_b", [4])]."""
    out = []
    for item in text.split(";"):
        if not item.strip():
            continue
        root, _, eps = item.strip().rpartition(":")
        if not root or not eps:
            raise ValueError(f"setrep_rebot_sets entries are root:ep,ep, got {item!r}")
        out.append((root.strip(), [int(e) for e in eps.split(",") if e.strip()]))
    return out


def parse_named(text):
    """``"hero=root:0:0-234@1;roll_c1=root:4:13-19@0.5"`` -> [{"label", "root", "episode", "t0", "t1", "stride_s"}]."""
    out = []
    for item in text.split(";"):
        if not item.strip():
            continue
        try:
            label, rest = item.strip().split("=", 1)
            spec, stride = rest.rsplit("@", 1)
            root, episode, window = spec.rsplit(":", 2)
            t0, t1 = window.split("-", 1)
            out.append({"label": label.strip(), "root": root.strip(), "episode": int(episode), "t0": float(t0),
                        "t1": float(t1), "stride_s": float(stride)})
        except ValueError as exc:
            raise ValueError(f"setrep_named entries are label=root:ep:t0-t1@stride_s, got {item!r}") from exc
    return out


def root_name(path):
    """Basename of a root; a raw ``inference_dataset`` takes its run folder's name."""
    path = Path(path).resolve()
    return path.parent.name if path.name == "inference_dataset" else path.name


def phase_verb(subtask):
    """"grasp the cup" -> "grasp"; move / release kept, anything else "other"."""
    verb = (subtask or "").strip().split(" ")[0].lower()
    return verb if verb in ("grasp", "move", "release") else "other"


def balanced_picks(verbs, k):
    """k positions over range(len(verbs)) split as evenly as the verbs present allow (one
    pick at a time to the verb with the fewest so far), each verb's share evenly spaced
    over its own frames."""
    buckets = {}
    for i, v in enumerate(verbs):
        buckets.setdefault(v, []).append(i)
    names = sorted(buckets, key=lambda v: -len(buckets[v]))
    share = {v: 0 for v in names}
    for _ in range(min(k, len(verbs))):
        v = min((v for v in names if share[v] < len(buckets[v])), key=lambda v: share[v])
        share[v] += 1
    picks = []
    for v in names:
        picks += [buckets[v][int(j)] for j in even_picks(len(buckets[v]), share[v])]
    return np.array(sorted(picks), dtype=int)


def frame_subtasks(dataset):
    """One subtask string per global frame index: the reviewed windows when the root has
    them, else the online labels of a raw inference_dataset, else empty strings."""
    from lerobot.probes.utils import _dataset_subtask_indices, get_subtask_str

    n = len(dataset)
    indices = _dataset_subtask_indices(dataset)
    if indices is not None:
        names = {}
        return [names.setdefault(int(i), get_subtask_str(dataset, int(i))) for i in indices.tolist()[:n]]
    labels = Path(str(dataset.root)) / "meta" / "online_labels.parquet"
    if labels.exists():
        import pandas as pd

        df = pd.read_parquet(labels, columns=["index", "subtask"])
        by_index = dict(zip(df["index"].astype(int), df["subtask"].astype(str)))
        return [by_index.get(i, "") for i in range(n)]
    return [""] * n


def even_picks(n, k):
    """k positions evenly spaced over range(n), the ends included, never repeated."""
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))


# ──────────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────────

def pick_episodes(episodes, task_of, n, rng):
    """n episodes by seed, every task once before any task repeats."""
    order = list(episodes)
    rng.shuffle(order)
    chosen, seen = [], set()
    for ep in order:
        if task_of[ep] not in seen and len(chosen) < n:
            chosen.append(ep)
            seen.add(task_of[ep])
    for ep in order:
        if len(chosen) < n and ep not in chosen:
            chosen.append(ep)
    return sorted(chosen)


class _Root:
    """One loaded LeRobot root: identity, per-frame columns, episodes, tasks, subtasks."""

    def __init__(self, dataset, cfg):
        from lerobot.datasets.embodiment import embodiment_name

        self.dataset = dataset
        self.identity = dataset_identity_columns(dataset, cfg)
        self.robot = embodiment_name(self.identity["embodiment_index"]) or f"embodiment {self.identity['embodiment_index']}"
        self.stride = probe_image_stride(cfg)
        self.columns = dataset.hf_dataset.select_columns(["frame_index", "timestamp", "task_index"]).with_format(None)[:]
        self.source = root_name(str(dataset.root))
        self.episodes = build_episode_index(dataset)
        self.task_of = {ep: str(dataset.meta.tasks.iloc[int(self.columns["task_index"][idx[0]])].name)
                        for ep, idx in self.episodes.items()}
        self.subtasks = frame_subtasks(dataset)

    def sample(self, set_name, ep, index, **extra):
        return {**self.identity, "kind": "lerobot", "set": set_name, "source": self.source, "robot": self.robot,
                "episode": f"{self.source}/{ep}", "index": int(index),
                "frame": int(self.columns["frame_index"][index]), "time_s": float(self.columns["timestamp"][index]),
                "task": self.task_of[ep], "subtask": self.subtasks[index], **extra}

    def episode_samples(self, set_name, ep, n_frames, balance):
        available = [i for i in self.episodes[ep] if i % self.stride == 0]
        if balance:
            picks = balanced_picks([phase_verb(self.subtasks[i]) for i in available], n_frames)
        else:
            picks = even_picks(len(available), n_frames)
        return [self.sample(set_name, ep, available[int(pos)]) for pos in picks]

    def window_samples(self, set_name, ep, t0, t1, stride_s, label):
        """Every stride-grid frame of the episode with timestamp in [t0, t1], at least
        stride_s apart (episode-relative seconds, as the dataset's timestamps are)."""
        base = float(self.columns["timestamp"][self.episodes[ep][0]])
        out, last = [], None
        for i in self.episodes[ep]:
            t = float(self.columns["timestamp"][i]) - base
            if i % self.stride or t < t0 or t > t1:
                continue
            if last is None or t - last >= stride_s - 1e-6:
                out.append(self.sample(set_name, ep, i, clip=label))
                last = t
        if not out:
            raise ValueError(f"No frames for named clip {label!r} in {self.source}/{ep} at {t0}-{t1} s")
        return out


def lerobot_set(dataset, cfg, set_name, n_episodes, n_frames, rng, balance=False, roots=None):
    """Random episode picks (every task once before any repeats); the loaded root goes into ``roots``."""
    root = dataset if isinstance(dataset, _Root) else _Root(dataset, cfg)
    if roots is not None:
        roots[root.source] = root
    samples = []
    for ep in pick_episodes(root.episodes, root.task_of, n_episodes, rng):
        samples += root.episode_samples(set_name, ep, n_frames, balance)
    return samples


def open_roots(specs, cfg, roots):
    """Load every root named in specs once (keyed by basename) into ``roots``."""
    for spec in specs:
        path = spec if isinstance(spec, str) else spec["root"]
        name = root_name(path)
        if name not in roots:
            roots[name] = _Root(load_extra_dataset(cfg.dataset.repo_id, path), cfg)
    return roots


def explicit_sets(spec_text, cfg, set_name, n_frames, balance, roots):
    samples = []
    for path, eps in parse_rebot_sets(spec_text):
        root = open_roots([path], cfg, roots)[root_name(path)]
        for ep in eps:
            samples += root.episode_samples(set_name, ep, n_frames, balance)
    return samples


def named_set(spec_text, cfg, roots):
    samples = []
    for spec in parse_named(spec_text):
        root = open_roots([spec], cfg, roots)[root_name(spec["root"])]
        samples += root.window_samples("named", spec["episode"], spec["t0"], spec["t1"], spec["stride_s"], spec["label"])
    return samples


BUFFER_OF_SPLIT = {"train": "diverse", "holdout": "diverse_holdout"}


def diverse_set(buffer, specs, n_frames, rng, split="train"):
    """Episodes per (source, embodiment) from a diverse buffer (training or holdout, the
    ``split`` tag), n_frames anchors evenly spaced over each: the listed episode ids, or
    random ones with at least n_frames anchors when the source has them, else its longest."""
    from lerobot.datasets.diverse_prompt import episode_task

    samples = []
    for source, embodiment, n_episodes in specs:
        by_episode = defaultdict(list)
        for index, row in enumerate(buffer.rows):
            if str(row["source"]) == source and str(row["embodiment"]) == embodiment:
                by_episode[str(row["episode_id"])].append(index)
        if not by_episode:
            raise ValueError(f"No diverse buffer rows for {source}:{embodiment}")
        anchors = {ep: list({float(buffer.rows[i]["anchor_s"]): i for i in sorted(
            idx, key=lambda i: float(buffer.rows[i]["anchor_s"]))}.values()) for ep, idx in by_episode.items()}
        if isinstance(n_episodes, list):
            missing = sorted(set(n_episodes) - set(anchors))
            if missing:
                raise ValueError(f"No diverse {split} buffer rows for {source}:{embodiment} episodes {missing}")
            chosen = n_episodes
        else:
            pool = sorted(ep for ep, idx in anchors.items() if len(idx) >= n_frames) \
                or [max(anchors, key=lambda ep: len(anchors[ep]))]
            chosen = [pool[int(i)] for i in rng.permutation(len(pool))[:n_episodes]]
        for ep in sorted(chosen):
            task = episode_task(buffer.selection.episode_records[ep], source)[0]
            for pos in even_picks(len(anchors[ep]), n_frames):
                index = anchors[ep][int(pos)]
                row = buffer.rows[index]
                samples.append({"kind": "diverse", "set": "diverse", "source": source, "robot": embodiment,
                                "episode": f"{source}/{ep}", "index": int(index), "frame": int(row["anchor_frame"]),
                                "time_s": float(row["anchor_s"]), "task": task, "split": split})
    return samples


def external_root(cfg):
    for source in getattr(cfg.dataset, "sources", None) or []:
        if str(source.name).startswith("external") and source.root:
            return str(source.root)
    raise ValueError("No dataset source named external* in the config")


def sampling_plan(val_dataset, cfg):
    """Samples in page order (set, episode, time), the datasets by source name, the buffers."""
    p = cfg.probe_parameters
    rng = np.random.default_rng(int(p.random_seed))
    balance = bool(getattr(p, "setrep_balance_phases", False))
    roots, samples = {}, {}
    if getattr(p, "setrep_rebot_sets", ""):
        samples["rebot"] = explicit_sets(p.setrep_rebot_sets, cfg, "rebot", p.setrep_rebot_frames, balance, roots)
    else:
        samples["rebot"] = lerobot_set(val_dataset, cfg, "rebot", p.setrep_rebot_episodes, p.setrep_rebot_frames, rng, balance, roots)
    if getattr(p, "setrep_external_sets", ""):
        samples["external"] = explicit_sets(p.setrep_external_sets, cfg, "external", p.setrep_external_frames, balance, roots)
    else:
        ext = load_extra_dataset(cfg.dataset.repo_id, external_root(cfg))
        samples["external"] = lerobot_set(ext, cfg, "external", p.setrep_external_episodes, p.setrep_external_frames, rng, balance, roots)
    buffers = {}
    specs = parse_sources(p.setrep_diverse_sources)
    holdout = parse_sources(getattr(p, "setrep_diverse_holdout", ""))
    if specs or holdout:
        buffers = _open_diverse_buffers(cfg)
        samples["diverse"] = (diverse_set(buffers["diverse"], specs, p.setrep_diverse_frames, rng, "train")
                              + diverse_set(buffers.get("diverse_holdout"), holdout, p.setrep_diverse_frames, rng, "holdout"))
    if getattr(p, "setrep_named", ""):
        samples["named"] = named_set(p.setrep_named, cfg, roots)
    datasets = {name: r.dataset for name, r in roots.items()}
    return [s for name in SETS for s in samples.get(name, [])], datasets, buffers


# ──────────────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(adapter, cfg, samples, datasets, buffers, cache_dir):
    layers, groups = _options(cfg)
    p = cfg.probe_parameters
    cache_dir = Path(cache_dir)
    (cache_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").unlink(missing_ok=True)
    arrays, rows = {}, []
    present = {g: np.zeros(len(samples), dtype=bool) for g in groups}
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            logging.info("Set representations: %s/%s frames", i + 1, len(samples))
        inputs = (_rebot_inputs(datasets[sample["source"]], cfg, sample, adapter.chunk_size)
                  if sample["kind"] == "lerobot" else _diverse_inputs(buffers[BUFFER_OF_SPLIT[sample["split"]]], cfg, sample))
        if not inputs["subtask"] and sample.get("subtask"):
            inputs["subtask"] = sample["subtask"]  # raw inference_dataset: the prompt the policy was given
        for camera in ("external_0", "wrist_0"):
            (cache_dir / "thumbs" / f"{i:04d}.{camera}.jpg").unlink(missing_ok=True)
        _save_thumbs(inputs["obs"], str(cache_dir), i)
        reps = adapter.capture_layer_representations(
            inputs["obs"], inputs["task"], subtask=inputs["subtask"], metadata=inputs["metadata"],
            extra_complementary=inputs["extra"], noise_seed=int(p.random_seed), output_dtype=torch.float32,
            capture_native_views=p.setrep_view != "block_output",
        )
        if p.setrep_view != "block_output":
            reps = reps.get("native_views", {}).get(p.setrep_view)
            if reps is None:
                raise ValueError(f"Adapter did not capture native view {p.setrep_view}")
        for site, group in GROUPS:
            if group not in groups:
                continue
            vec = reps.get(site, {}).get(group)
            if vec is None:
                continue
            if max(layers) >= vec.shape[0]:
                raise ValueError(f"setrep_layers={layers} exceeds {group}'s {vec.shape[0]} blocks")
            values = vec[layers].detach().float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite representation at sample {i}/{group}")
            if group not in arrays:
                arrays[group] = np.lib.format.open_memmap(cache_dir / f"{group}.npy", mode="w+", dtype=np.float32,
                                                          shape=(len(samples), *values.shape))
            arrays[group][i] = values
            present[group][i] = True
        rows.append({**sample, "row": i, "task": inputs["task"], "subtask": inputs["subtask"]})
    if not arrays:
        raise ValueError("No requested token groups were captured")
    for array in arrays.values():
        array.flush()
    for group, mask in present.items():
        np.save(cache_dir / f"{group}.present.npy", mask)
    meta = {"protocol": PROTOCOL, "view": p.setrep_view, "layers": layers, "groups": list(arrays),
            "requested_groups": groups, "rows": rows, "noise_seed": int(p.random_seed),
            "checkpoint": str(getattr(cfg.policy, "pretrained_path", "")), "flow_time": 0}
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_cache(cache_dir):
    root = Path(cache_dir)
    if not (root / "meta.json").exists():
        raise ValueError("No set representation cache; collect first")
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible set representation cache protocol; collect again")
    arrays, present = {}, {}
    for group in meta["groups"]:
        arrays[group] = np.load(root / f"{group}.npy", mmap_mode="r")
        present[group] = np.load(root / f"{group}.present.npy")
        if arrays[group].shape[:2] != (len(meta["rows"]), len(meta["layers"])) or present[group].shape != (len(meta["rows"]),):
            raise ValueError(f"Incomplete cache for {group}; collect again")
    return meta, arrays, present


# ──────────────────────────────────────────────────────────────────────────────
# Analysis (CPU)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_matrix(x, centered):
    x = np.asarray(x, dtype=np.float64)
    if centered:
        x = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(x, axis=1)
    z = x / np.where(norms > 1e-8, norms, 1.0)[:, None]
    return np.clip(z @ z.T, -1.0, 1.0), norms


def distance_matrix(cosine, norms):
    """L2 distance from the cosine and the two norms, as the page derives it."""
    n = np.asarray(norms)[:, None]
    return np.sqrt(np.maximum(n ** 2 + n.T ** 2 - 2 * n * n.T * cosine, 0.0))


def block_means(cosine, labels, episodes, exclude="episode"):
    """Mean cosine between every pair of label groups. ``exclude="episode"`` drops pairs of
    frames from the same episode (set blocks); ``exclude="self"`` drops only a frame with
    itself (episode blocks). Returns the label names in first-seen order, the mean matrix
    (nan where no pair) and the pair counts."""
    names = list(dict.fromkeys(labels))
    lab, ep = np.asarray(labels), np.asarray(episodes)
    other = ep[:, None] != ep[None, :] if exclude == "episode" else ~np.eye(len(ep), dtype=bool)
    means = np.full((len(names), len(names)), np.nan)
    counts = np.zeros((len(names), len(names)), dtype=int)
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            mask = (lab[:, None] == na) & (lab[None, :] == nb) & other
            counts[a, b] = int(mask.sum())
            if counts[a, b]:
                means[a, b] = float(cosine[mask].mean())
    return names, means, counts


def set_gaps(names, means, measure="cosine"):
    """Cosine: mean of the two within-set means minus the between-set mean. Distance: the
    between-set mean minus the within-set average. Positive = more alike within than between."""
    gaps = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            within = (means[a, a] + means[b, b]) / 2
            if np.isfinite(within) and np.isfinite(means[a, b]):
                gap = within - means[a, b] if measure == "cosine" else means[a, b] - within
                gaps[f"{names[a]}_{names[b]}"] = {"within_a": means[a, a], "within_b": means[b, b],
                                                  "between": means[a, b], "gap": gap}
    return gaps


def _nan_to_none(values):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in values]


def analyze(meta, arrays, present, layers, groups, output_dir):
    root = Path(output_dir)
    (root / "panels").mkdir(parents=True, exist_ok=True)
    all_rows = meta["rows"]
    blocks_csv, episode_csv, panel_keys = [], [], []
    summary_gaps = {m: {g: {} for g in GEOMETRIES} for m in MEASURES}
    for group in groups:
        if group not in arrays:
            logging.warning("Set representation group %s was absent at capture", group)
            continue
        ids = np.flatnonzero(present[group])
        if len(ids) < 2:
            continue
        rows = [all_rows[i] for i in ids]
        sets = [r["set"] for r in rows]
        episodes = [r["episode"] for r in rows]
        for layer in layers:
            li = meta["layers"].index(layer)
            x = np.asarray(arrays[group][ids, li], dtype=np.float64)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite cached vectors in {group}/L{layer}")
            for geometry in GEOMETRIES:
                logging.info("Analyze sets: %s L%s %s", group, layer, geometry)
                cosine, norms = cosine_matrix(x, geometry == "centered")
                matrices = {"cosine": cosine, "distance": distance_matrix(cosine, norms)}
                key = f"{group}.L{layer}.{geometry}"
                header = {"group": group, "layer": layer, "geometry": geometry}
                blocks = {}
                for measure, matrix in matrices.items():
                    ep_names, ep_means, ep_counts = block_means(matrix, episodes, episodes, exclude="self")
                    set_names, set_means, set_counts = block_means(matrix, sets, episodes)
                    gaps = set_gaps(set_names, set_means, measure)
                    for a, na in enumerate(set_names):
                        for b, nb in enumerate(set_names):
                            blocks_csv.append({**header, "measure": measure, "set_a": na, "set_b": nb,
                                               "mean": set_means[a, b], "pairs": set_counts[a, b]})
                    for a, na in enumerate(ep_names):
                        for b, nb in enumerate(ep_names):
                            episode_csv.append({**header, "measure": measure, "episode_a": na, "episode_b": nb,
                                                "mean": ep_means[a, b], "pairs": ep_counts[a, b]})
                    summary_gaps[measure][geometry].setdefault(group, {})[f"L{layer}"] = {k: round(v["gap"], 4) for k, v in gaps.items()}
                    blocks[measure] = {"episode": {"names": ep_names, "means": _nan_to_none(ep_means), "counts": ep_counts.tolist()},
                                       "set": {"names": set_names, "means": _nan_to_none(set_means), "counts": set_counts.tolist()},
                                       "gaps": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in gaps.items()}}
                data = {"ids": ids.tolist(), "cosine": np.round(cosine, 3).tolist(),
                        "norms": np.round(norms, 3).tolist(), "blocks": blocks}
                (root / "panels" / f"{key}.js").write_text(
                    f"window.__setPanel({json.dumps(key)},{json.dumps(data, separators=(',', ':'))});")
                panel_keys.append(key)
    if not panel_keys:
        raise ValueError("No selected group has enough captured frames to analyze")
    _write_csv(root / "blocks.csv", blocks_csv)
    _write_csv(root / "episode_blocks.csv", episode_csv)
    head = f"L{max(layers)}"
    summary = {"frames": len(all_rows), "episodes": len({r["episode"] for r in all_rows}),
               "sets": len({r["set"] for r in all_rows}),
               "per_set": {s: {"episodes": len({r["episode"] for r in all_rows if r["set"] == s}),
                               "frames": sum(r["set"] == s for r in all_rows)} for s in dict.fromkeys(r["set"] for r in all_rows)},
               "view": meta["view"], "headline_layer": head, "gap": summary_gaps["cosine"],
               "gap_distance": summary_gaps["distance"],
               "missing_groups": sorted(set(groups) - set(arrays))}
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    payload = {"rows": [{k: r.get(k) for k in ("row", "set", "source", "robot", "episode", "time_s", "frame", "task", "subtask", "clip")}
                        for r in all_rows],
               "panels": panel_keys, "groups": [g for g in groups if g in arrays], "layers": layers,
               "geometries": list(GEOMETRIES), "measures": list(MEASURES), "view": meta["view"],
               "checkpoint": meta.get("checkpoint", "")}
    return payload, summary


def _write_csv(path, records):
    if not records:
        return
    keys = list(dict.fromkeys(k for r in records for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


# ──────────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────────

_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Set representations</title>
<script>__PLOTLY__</script>
<style>
body{font-family:system-ui,sans-serif;margin:0;padding:0 16px 16px;background:#f4f6f9;color:#202631}
h1{font-size:18px;margin:14px 0 4px}h2{font-size:14px;margin:0 0 6px}.muted{color:#667085;font-size:12px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:8px 0}select,input,button{font-size:13px;padding:3px 6px}input[type=number]{width:80px}
.box{background:white;border:1px solid #e0e4eb;border-radius:8px;padding:12px;min-width:0;margin-top:14px}
.grid{display:grid;grid-template-columns:minmax(520px,3fr) minmax(320px,2fr);gap:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:1000px){.grid,.grid2{grid-template-columns:1fr}}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:10px}.pair img{width:100%;border-radius:4px;background:#ddd}
.card{font-size:12px}.card b{font-size:13px}.tag{display:inline-block;padding:1px 6px;border-radius:3px;color:white;font-size:11px;margin-right:4px}
.rebot{background:#2a6fdb}.external{background:#e37525}.diverse{background:#2e9e5b}.named{background:#8e44ad}
table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:3px 6px;border-bottom:1px solid #eee;text-align:right}th:first-child,td:first-child{text-align:left}
details{font-size:12px;color:#444;margin-top:8px}
</style></head><body>
<h1>Set representations</h1><p class="muted" id="info"></p>
<div class="controls">
 group <select id="group"></select> layer <select id="layer"></select> geometry <select id="geometry"></select> measure <select id="measure"></select>
 color range <input id="zlo" type="number" step="0.01"> to <input id="zhi" type="number" step="0.01"> <button id="zauto">auto</button>
 <span class="muted" id="loading"></span>
</div>
<div class="grid">
 <div class="box"><h2>Every frame against every frame, rows ordered set · episode · time</h2><div id="matrix" style="height:620px"></div>
 <p class="muted">Thin lines separate episodes, thick lines separate sets. Hover or click a cell to see its two frames on the right.</p></div>
 <div class="box"><h2>Selected pair</h2><div id="pair"><p class="muted">Hover the matrix.</p></div></div>
</div>
<div class="grid2">
 <div class="box"><h2>Episode blocks: mean over frame pairs (self excluded)</h2><div id="epblocks" style="height:420px"></div></div>
 <div class="box"><h2>Set blocks: mean over pairs of frames from different episodes</h2><div id="setblocks" style="height:260px"></div><div id="gaps"></div></div>
</div>
<details><summary>How to read</summary><p><b>Measure</b>: cosine compares directions, L2 distance compares direction and magnitude (derived from the stored cosine and the two frames' norms in the chosen geometry; small distance = alike, so its color scale is reversed and its gap is between minus within). <b>Raw</b> cosines between hidden states sit near 1 because every state shares one common direction; <b>centered</b> removes the mean over all sampled frames of this group and layer once, for every set together, so what is left is how the frames differ from each other. A set's own block on the diagonal is how alike two different episodes of that set are; an off-diagonal block is how alike frames from two sets are. The <b>gap</b> for a set pair is the average of their two within-set means minus their between-set mean: near 0, the policy does not tell the sets apart in this group at this depth; near the within-set value, they never meet. The subtask and state groups separate sets partly by prompt construction (different task text, state width); the image groups and the action tokens are where a shared or split representation is informative.</p></details>
<script>
const DATA=__DATA__;const $=id=>document.getElementById(id);const panels={};let waiting=null;
window.__setPanel=(key,data)=>{panels[key]=data;if(waiting&&waiting.key===key){waiting.resolve(data);waiting=null;}};
function loadPanel(key){if(panels[key])return Promise.resolve(panels[key]);return new Promise(resolve=>{waiting={key,resolve};const s=document.createElement('script');s.src='panels/'+key+'.js';document.body.append(s);});}
const SETCOLOR={rebot:'#2a6fdb',external:'#e37525',diverse:'#2e9e5b',named:'#8e44ad'};
function fill(id,values,labels){const s=$(id);s.replaceChildren();values.forEach((v,i)=>{const o=document.createElement('option');o.value=v;o.textContent=labels?labels[i]:v;s.append(o);});}
fill('group',DATA.groups);fill('layer',DATA.layers,DATA.layers.map(l=>'L'+l));fill('geometry',DATA.geometries);fill('measure',DATA.measures,['Cosine','L2 distance']);
$('layer').value=DATA.layers[DATA.layers.length-1];$('geometry').value='centered';
const q=new URLSearchParams(location.hash.slice(1));for(const k of ['group','layer','geometry','measure'])if(q.get(k))$(k).value=q.get(k);
// One color range for all three heatmaps. Auto = data range of the current panel; typing a limit pins it until auto is pressed.
let manualRange=q.get('zlo')!=null&&q.get('zhi')!=null;if(manualRange){$('zlo').value=q.get('zlo');$('zhi').value=q.get('zhi');}
function colorRange(lo,hi,centered){if(!manualRange){const m=Math.max(Math.abs(lo),Math.abs(hi));$('zlo').value=(centered?-m:lo).toFixed(3);$('zhi').value=(centered?m:hi).toFixed(3);}return [Number($('zlo').value),Number($('zhi').value)];}
// One stored cosine serves both measures: distance follows from the two norms.
function cell(P,i,j){const c=P.cosine[i][j];if($('measure').value==='cosine')return c;const a=P.norms[i],b=P.norms[j];return Math.sqrt(Math.max(0,a*a+b*b-2*a*b*c));}
$('info').textContent=`${DATA.rows.length} frames · view ${DATA.view} · ${DATA.checkpoint}`;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt=(v,d=3)=>v==null||!Number.isFinite(v)?'—':v.toFixed(d);
function card(r){const n=String(r.row).padStart(4,'0');return `<div class="card"><span class="tag ${r.set}">${esc(r.set)}</span><b>${esc(r.source)}</b> · ${esc(r.episode.split('/').pop())} · ${fmt(r.time_s,1)} s<br><span class="muted">${esc(r.robot)} · ${esc(r.task)}<br>${esc(r.subtask)}</span><div class="pair"><img src="cache/thumbs/${n}.external_0.jpg" onerror="this.style.visibility='hidden'"><img src="cache/thumbs/${n}.wrist_0.jpg" onerror="this.style.visibility='hidden'"></div></div>`;}
function showPair(P,i,j){const a=DATA.rows[P.ids[i]],b=DATA.rows[P.ids[j]];$('pair').innerHTML=`<p class="muted" style="margin:0 0 6px">cosine ${fmt(P.cosine[i][j])} · L2 ${fmt(Math.sqrt(Math.max(0,P.norms[i]**2+P.norms[j]**2-2*P.norms[i]*P.norms[j]*P.cosine[i][j])))}</p><div class="pair">${card(a)}${card(b)}</div>`;}
function boundaries(rows){const eps=[],sets=[];for(let i=1;i<rows.length;i++){if(rows[i].episode!==rows[i-1].episode)eps.push(i-0.5);if(rows[i].set!==rows[i-1].set)sets.push(i-0.5);}return {eps,sets};}
function centers(rows,keyOf){const out=[];let start=0;for(let i=1;i<=rows.length;i++){if(i===rows.length||keyOf(rows[i])!==keyOf(rows[start])){out.push({pos:(start+i-1)/2,label:keyOf(rows[start])});start=i;}}return out;}
async function update(){const g=$('group').value,l=$('layer').value,geo=$('geometry').value,meas=$('measure').value;const key=`${g}.L${l}.${geo}`;const h={group:g,layer:l,geometry:geo,measure:meas};if(manualRange){h.zlo=$('zlo').value;h.zhi=$('zhi').value;}location.hash=new URLSearchParams(h).toString();
 $('loading').textContent='loading…';const P=await loadPanel(key);$('loading').textContent='';
 const rows=P.ids.map(i=>DATA.rows[i]);const b=boundaries(rows);const n=rows.length;
 const Z=P.cosine.map((row,i)=>row.map((_,j)=>cell(P,i,j)));
 const flat=Z.flat().filter(v=>v!=null);const lo=Math.min(...flat),hi=Math.max(...flat);
 const dist=meas==='distance';const centered=geo==='centered'&&!dist;const [zlo,zhi]=colorRange(lo,hi,centered);
 const scale=dist?'Viridis':centered?'RdBu':'Viridis';const label=dist?'L2 distance':'cosine';
 const ticks=centers(rows,r=>r.set+' · '+r.episode.split('/').pop());
 const shapes=[...b.eps.map(p=>({type:'line',x0:p,x1:p,y0:-0.5,y1:n-0.5,line:{color:'white',width:1}})),...b.eps.map(p=>({type:'line',y0:p,y1:p,x0:-0.5,x1:n-0.5,line:{color:'white',width:1}})),
  ...b.sets.map(p=>({type:'line',x0:p,x1:p,y0:-0.5,y1:n-0.5,line:{color:'black',width:3}})),...b.sets.map(p=>({type:'line',y0:p,y1:p,x0:-0.5,x1:n-0.5,line:{color:'black',width:3}}))];
 Plotly.react('matrix',[{type:'heatmap',z:Z,colorscale:scale,reversescale:dist,zmin:zlo,zmax:zhi,zauto:false,hovertemplate:'row %{y} · col %{x}<br>'+label+' %{z:.3f}<extra></extra>'}],
  {margin:{l:150,r:10,t:10,b:120},xaxis:{tickvals:ticks.map(t=>t.pos),ticktext:ticks.map(t=>t.label),tickangle:-45,tickfont:{size:10},constrain:'domain'},yaxis:{tickvals:ticks.map(t=>t.pos),ticktext:ticks.map(t=>t.label),tickfont:{size:10},autorange:'reversed',scaleanchor:'x'},shapes},{responsive:true});
 const m=$('matrix');m.removeAllListeners?.('plotly_hover');m.removeAllListeners?.('plotly_click');
 const onCell=e=>{const p=e.points[0];if(p)showPair(P,p.y,p.x);};m.on('plotly_hover',onCell);m.on('plotly_click',onCell);
 const B=P.blocks[meas];const E=B.episode;const eLabels=E.names.map(name=>{const r=rows.find(r=>r.episode===name);return r.set+' · '+name.split('/').pop();});
 Plotly.react('epblocks',[{type:'heatmap',z:E.means,x:eLabels,y:eLabels,text:E.means.map(r=>r.map(v=>fmt(v,2))),texttemplate:'%{text}',colorscale:scale,reversescale:dist,zmin:zlo,zmax:zhi,zauto:false,hovertemplate:'%{y}<br>%{x}<br>mean '+label+' %{z}<extra></extra>'}],
  {margin:{l:160,r:10,t:10,b:130},xaxis:{tickangle:-45,tickfont:{size:10}},yaxis:{autorange:'reversed',tickfont:{size:10}}},{responsive:true});
 const S=B.set;
 Plotly.react('setblocks',[{type:'heatmap',z:S.means,x:S.names,y:S.names,text:S.means.map(r=>r.map(v=>fmt(v,3))),texttemplate:'%{text}',colorscale:scale,reversescale:dist,zmin:zlo,zmax:zhi,zauto:false,hovertemplate:'%{y} vs %{x}<br>mean '+label+' %{z}<extra></extra>'}],
  {margin:{l:80,r:10,t:10,b:40},yaxis:{autorange:'reversed'}},{responsive:true});
 $('gaps').innerHTML='<table><thead><tr><th>Set pair</th><th>Within A</th><th>Within B</th><th>Between</th><th>Gap</th></tr></thead><tbody>'+Object.entries(B.gaps).map(([k,v])=>`<tr><td>${esc(k.replace('_',' vs '))}</td><td>${fmt(v.within_a)}</td><td>${fmt(v.within_b)}</td><td>${fmt(v.between)}</td><td><b>${fmt(v.gap)}</b></td></tr>`).join('')+'</tbody></table>';
}
for(const k of ['group','layer','geometry','measure'])$(k).onchange=update;
for(const k of ['zlo','zhi'])$(k).onchange=()=>{manualRange=true;update();};
$('zauto').onclick=()=>{manualRange=false;update();};
update();
</script></body></html>
"""


def write_report(payload, output_dir):
    from plotly.offline import get_plotlyjs

    html = _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", json.dumps(payload).replace("</", "<\\/"))
    (Path(output_dir) / "sets.html").write_text(html)


def refresh_report(path):
    """Re-render a stored sets.html from its cache when this renderer is newer than the page.

    The viewers call this before serving, so old probe folders pick up template changes
    without a manual re-render. Layers and groups come from the cache's own metadata.
    """
    path = Path(path)
    root = path.parent
    meta_path = root / "cache" / "meta.json"
    if path.name != "sets.html" or root.name != "set_representations" or not meta_path.exists():
        return
    if path.exists() and path.stat().st_mtime >= Path(__file__).stat().st_mtime:
        return
    meta, arrays, present = load_cache(root / "cache")
    payload, _ = analyze(meta, arrays, present, meta["layers"], meta["requested_groups"], root)
    write_report(payload, root)


# ──────────────────────────────────────────────────────────────────────────────
# Entry points
# ──────────────────────────────────────────────────────────────────────────────

def run(adapter, val_dataset, cfg, output_dir):
    p = cfg.probe_parameters
    layers, groups = _options(cfg)
    if p.mode not in ("collect", "plot", "all"):
        raise ValueError(f"Unknown probe mode {p.mode}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if p.mode in ("collect", "all"):
        if adapter is None:
            raise ValueError("Set representation collection requires a policy adapter")
        samples, datasets, buffers = sampling_plan(val_dataset, cfg)
        plan = {"protocol": PROTOCOL, "samples": samples, "frames": len(samples),
                "episodes": sorted({s["episode"] for s in samples})}
        (root / "sampling_plan.json").write_text(json.dumps(plan, indent=2))
        collect(adapter, cfg, samples, datasets, buffers, root / "cache")
    if p.mode == "collect":
        return None
    meta, arrays, present = load_cache(root / "cache")
    if p.setrep_view != meta["view"] or set(layers) - set(meta["layers"]) or set(groups) - set(meta["requested_groups"]):
        raise ValueError("Requested view/layers/groups differ from cache; collect again or select cached values")
    payload, summary = analyze(meta, arrays, present, layers, groups, root)
    write_report(payload, root)
    head = summary["headline_layer"]
    metrics = [Metric("frames", "Frames", fmt=0), Metric("episodes", "Episodes", fmt=0)]
    for group in HEADLINE_GROUPS:
        if group not in payload["groups"]:
            continue
        for pair, label in (("rebot_external", "ReBot vs external"), ("rebot_diverse", "ReBot vs diverse"),
                            ("external_diverse", "external vs diverse")):
            metrics.append(Metric(f"gap.centered.{group}.{head}.{pair}", f"{label} gap, {group} {head} (centered)",
                                  good="none", fmt=3, primary=pair != "external_diverse",
                                  trend=pair == "rebot_diverse",
                                  note="Within-set mean cosine over different episodes minus between-set mean; 0 = the sets are not told apart here."))
    write_index(str(root), sys.modules[__name__], title="Set representations", group="Representation",
                claim="Are our robot's episodes, the external ReBot recordings and the diverse robots represented in one region or apart, and where do they merge?",
                summary=summary, status="info", metrics=metrics,
                panels=[Panel("sets.html", "Frame-by-frame cosine or L2 matrix over every set with the frames on its axes, the episode block means, the set block means and the gaps, per group, layer, geometry and measure",
                              how="Pick centered geometry and the deepest layer. Thick lines are set borders: a matrix that is bright inside every set block and dark between them is a split representation, one that looks the same across the thick lines is shared. Read the gap table for the number; hover a cell to see which two frames it compares.",
                              primary=True),
                        Panel("blocks.csv", "Set-by-set mean cosine and mean L2 distance over pairs of frames from different episodes, per group, layer and geometry"),
                        Panel("episode_blocks.csv", "Episode-by-episode mean cosine and mean L2 distance, per group, layer and geometry"),
                        Panel("sampling_plan.json", "The exact episodes and frames of every set")],
                see_also=["representations", "domain_representations", "conditions_matrix"])
    return summary


@parser.wrap()
def probe_cli(cfg: ProbeSetRepresentationsConfig):
    init_logging()
    adapter = dataset = None
    if cfg.probe_parameters.mode in ("collect", "all"):
        val_root = getattr(cfg, "val_dataset_path", None)
        if not val_root:
            raise ValueError("set_representations needs val_dataset_path for the rebot set")
        dataset = load_extra_dataset(cfg.dataset.repo_id, str(val_root))
        device = get_safe_torch_device(try_device=cfg.policy.device)
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    step = next((int(part) for part in reversed(Path(str(cfg.policy.pretrained_path or "")).parts) if part.isdigit()), 0)
    run(adapter, dataset, cfg, Path(cfg.probe_parameters.output_dir) / "validation" / f"step_{step:08d}" / "set_representations")


def main():
    register_config_choices()
    from lerobot.scripts.rl_offline import _extract_config_path_args, _preprocess_config_yaml

    config_path, remaining = _extract_config_path_args(sys.argv[1:])
    if config_path:
        sys.argv = [sys.argv[0], *remaining, f"--config_path={_preprocess_config_yaml(config_path)}"]
    probe_cli()


if __name__ == "__main__":
    main()
