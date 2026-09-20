#!/usr/bin/env python
"""Conditions matrix: are objects and phases represented the same way across robots?

**Question.** Does the policy arrange bottle, cup and cloth, at grasp, move and release,
in the same relational geometry on ReBot as on the corpus robots? Domain separability
answers nothing here: any two robots' frames are separable by camera and prompt alone.
The offset-free question is whether the *structure over conditions* agrees.

**Conditions.** A cell is an object class x a phase. Object classes bottle, cup, cloth,
tape, block come from the noun of the ReBot subtask window or of the corpus atom; the
phases are grasp, move, release (return carries no object). ReBot additionally keeps the
object INSTANCE (spray bottle, pill bottle, cup, tape roll, sock, shirt) for its own
matrix. Robots: rebot, droid (droid + droid_success), molmoact, robochallenge. A cell
exists on a robot when at least two of its episodes carry it. Per (episode, cell) at
most ``conditions_frames_per_episode_cell`` frames, evenly spaced over the window, and at
most ``conditions_episodes_per_cell`` episodes per cell: every cell is an episode-balanced
sample, and the frame rate is irrelevant because every similarity below is between
frames of DIFFERENT episodes.

**Text conditions.** Every frame is captured twice: with its real task and subtask text,
and with neutral text that names no object ("Put the objects in the containers." /
"grasp the object"). The subtask clause under real text is the language ceiling, where
the matrices must agree because the words do. Object structure that survives the neutral
text came through the images.

**Vectors.** Per layer and token group, the pooled hidden state from the adapter seam
``capture_layer_representations`` (flow time t = 0, one fixed noise draw for every frame).
Groups: img_wrist_0, img_external_0, subtask, action_output (encoder) and action (expert).
Per robot the vectors are centred on that robot's training-frame mean and unit-normalised,
so the between-robot offset never enters.

**Matrix.** For robot $A$ with cells $1..K$,

$$M_A[i,j] = \\frac{1}{|P_{ij}|}\\sum_{(a,b)\\in P_{ij}} \\langle h_a, h_b\\rangle ,$$

$P_{ij}$ the pairs with $a$ in cell $i$, $b$ in cell $j$ and $\\text{episode}(a) \\neq
\\text{episode}(b)$. Off the diagonal this is the inner product of cell means with
same-episode pairs removed; on the diagonal it is the cell's cross-episode consistency.

**Score.** Spearman $\\rho_{AB}$ over the strict upper triangle of the cells both robots
have. Null: 2,000 permutations of $B$'s cell labels, reported as the 95th percentile and
a p-value. Ceiling: split each robot's episodes in half, $\\rho$ between the two half
matrices, $\\sqrt{\\rho_{AA}\\rho_{BB}}$ averaged over 20 splits. Reported:
$\\rho_{AB}$ / ceiling, the fraction of the reliable structure that is shared.

**Organisation.** The upper triangle regressed on same-class and same-phase indicators
(plus same-instance on ReBot's instance matrix): standardised weights per robot.

**Cross-robot decoding.** The matrix is invariant to a rotation of each robot's space, so
matching structure is consistent with parallel, disjoint codes. The direct test: for every
pair and both directions, ``dst``'s frames decoded by nearest cosine to ``src``'s cell
means over the cells both robots have, each robot centred on the mean of its shared-cell
means (a different cell mix adds no offset). Balanced accuracy for the cell, the object
class alone and the phase alone (chance 1/K). Ceiling: the same frames decoded with their
own robot's leave-one-episode-out cell means. Cross at the ceiling = the same directions
carry the code; cross at chance with the ceiling high = separate codes with the same shape.

**Held out.** ReBot: the validation set's windows. Diverse: ``<root>/holdout_episodes.json``
(never trained on, decoded from video, see ``holdout_actor_selection``). Each held-out
frame is scored by inner product with its robot's training cell means: nearest-cell
accuracy, and the Spearman between its similarity row and the training matrix's row for
its true cell.

**Output** (``<output_dir>/conditions_matrix/``): ``sharing_by_layer.png`` (rho / ceiling
against depth, one panel per robot pair x text condition, null line), ``matrices.png``
(every robot's class-level matrix at the peak layer, action and wrist groups),
``rebot_instances.png`` (ReBot's instance-level matrix at the peak layer),
``matrices_L<n>.png`` (the same at ``conditions_layers``, default 14, 28 and 32),
``decoding.png`` (ReBot pairs, both directions: cross-robot class and phase accuracy
against the within-robot ceiling and chance, action and wrist groups, both texts, at the
headline and ``conditions_layers`` layers) with ``decoding.csv`` behind it,
``frames_<robot>.png`` (per robot, the frames behind its matrix: every cell, the first
window frame of one episode and the last of another, external over wrist, captioned with
the real subtask), ``metrics.csv`` / ``organisation.csv`` / ``holdout.csv`` /
``summary.json``, and ``cache/`` (one fp16 memmap per group plus meta.json and
``thumbs/`` with both camera views of every frame; ``--probe_parameters.mode=plot``
re-analyses it). Matrix row labels carry the frames / episodes each cell rests on.

Standalone (checkpoint, corpus and validation set from the config):

    uv run python -m lerobot.probes.conditions_matrix --config config_rl.yaml \\
        --probe_parameters.enable_conditions_matrix=true \\
        --probe_parameters.conditions_rebot_roots=outputs/bottle_grasping-train-annotated-v2,outputs/rebot_socks_basket-annotated-v3
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.probes.base import ProbablePolicy
from lerobot.probes.domain_representations import _diverse_inputs
from lerobot.probes.manifest import Metric, Panel, write_index
from lerobot.probes.utils import (
    DEPLOYMENT_METADATA,
    as_image,
    build_episode_index,
    dataset_identity_columns,
    load_extra_dataset,
    makedirs,
    probe_frame_inputs,
    probe_image_stride,
    register_config_choices,
)
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.utils import init_logging


@dataclass
class ProbeConditionsMatrixConfig(TrainRLServerPipelineConfig):
    """Tunables under ``cfg.probe_parameters`` (ProbeConfig)."""


REBOT = "rebot"
ROBOT_OF_SOURCE = {
    "droid": "droid", "droid_success": "droid", "molmoact": "molmoact",
    "robochallenge": "robochallenge", "ur7e": "ur7e",
}
ROBOT_ORDER = (REBOT, "droid", "molmoact", "robochallenge", "ur7e")
PHASES = ("grasp", "move", "release")
CLASSES = ("bottle", "cup", "cloth", "tape", "block")
# Noun -> object class. "paper towel" is paper, not cloth.
CLASS_PATTERNS = (
    (r"\bpaper towels?\b", None),
    (r"\bbottles?\b", "bottle"), (r"\bcanisters?\b", "bottle"),
    (r"\bmugs?\b", "cup"), (r"\bcups?\b", "cup"),
    (r"\btowels?\b", "cloth"), (r"\bcloths?\b", "cloth"), (r"\bt?-?shirts?\b", "cloth"),
    (r"\bsocks?\b", "cloth"), (r"\brags?\b", "cloth"),
    (r"\btape\b", "tape"), (r"\bblocks?\b", "block"),
)
# (site, group) read out of the capture.
GROUPS = (
    ("encoder", "img_wrist_0"), ("encoder", "img_external_0"), ("encoder", "subtask"),
    ("encoder", "action_output"), ("action_expert", "action"),
)
GROUP_NAMES = tuple(g for _, g in GROUPS)
TEXT_CONDITIONS = ("real", "neutral")
NEUTRAL_TASK = "Put the objects in the containers."
NEUTRAL_SUBTASK = {
    "grasp": "grasp the object",
    "move": "move the object to the container",
    "release": "release the object in the container",
}
N_NULL = 2000
N_SPLITS = 20
MIN_EPISODES_PER_CELL = 2
MIN_SHARED_CELLS = 4
THUMB_CAMERAS = ("external_0", "wrist_0")
THUMB_PX = 256
N_SHOW = 2  # episodes shown per cell in frames_<robot>.png


# ──────────────────────────────────────────────────────────────────────────────
# Conditions
# ──────────────────────────────────────────────────────────────────────────────

def _parse_subtask(text: str) -> tuple[str, str, str | None] | None:
    """'move the black spray bottle to the bin' -> ('move', 'black spray bottle', 'bin')."""
    m = re.match(r"^(\S+)\s+(?:the\s+)?(.*?)(?:\s+(?:to|in|into|on)\s+(?:the\s+)?(.*))?$", text.strip().lower())
    if m is None:
        return None
    return m.group(1), m.group(2).strip(), (m.group(3).strip() if m.group(3) else None)


def _object_class(noun: str) -> str | None:
    noun = noun.lower()
    for pattern, cls in CLASS_PATTERNS:
        if re.search(pattern, noun):
            return cls
    return None


def _rebot_instance(noun: str) -> str:
    for key, name in (("spray", "spray bottle"), ("pill", "pill bottle"), ("sock", "sock"), ("shirt", "shirt")):
        if key in noun:
            return name
    return noun


def _cap_episodes(episodes: list, cap: int, rng) -> list:
    if len(episodes) <= cap:
        return episodes
    return sorted(rng.choice(episodes, size=cap, replace=False).tolist())


def _even_picks(n_available: int, n_wanted: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_available - 1, min(n_wanted, n_available), dtype=int))


def _rebot_samples(dataset, cfg, name: str, holdout: bool, rng) -> list[dict]:
    """One sample per chosen frame of every (instance, phase) window of a ReBot root."""
    p = cfg.probe_parameters
    identity = dataset_identity_columns(dataset, cfg)
    windows = json.load(open(os.path.join(dataset.root, "meta", "subtask_windows.json")))["episodes"]
    ep_index = build_episode_index(dataset)
    stride = probe_image_stride(cfg)
    cells: dict[tuple[str, str, str], dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for ep_key, wins in windows.items():
        ep = int(ep_key)
        if ep not in ep_index:
            continue
        for w in wins:
            parsed = _parse_subtask(w["subtask"])
            if parsed is None:
                continue
            verb, noun, dest = parsed
            cls = _object_class(noun)
            if verb not in PHASES or cls is None:
                continue
            # Window bounds are GLOBAL dataset indices, and the training cache selects its
            # image/depth rows on the global grid (load_depth_png tolerates the phase).
            lo, hi = int(w["from_index"]), int(w["to_index"])
            for g in range(lo, hi):
                if g % stride == 0:
                    cells[(_rebot_instance(noun), cls, verb)][ep].append((g, (g - lo) / max(hi - lo, 1), w["subtask"], dest))
    samples = []
    for (inst, cls, verb), by_ep in sorted(cells.items()):
        episodes = sorted(by_ep) if holdout else _cap_episodes(sorted(by_ep), p.conditions_episodes_per_cell, rng)
        for ep in episodes:
            frames = by_ep[ep]
            first = ep_index[ep][0]
            for k in _even_picks(len(frames), p.conditions_frames_per_episode_cell):
                global_idx, pos, text, dest = frames[int(k)]
                f = global_idx - first
                if int(dataset.hf_dataset[global_idx]["frame_index"].item()) != f:
                    raise ValueError(f"{name} episode {ep}: global index {global_idx} is not frame {f} of the episode")
                samples.append({
                    **identity,
                    "kind": "rebot", "source_key": name, "robot": REBOT, "source": name,
                    "episode": f"{name}/{ep}", "index": int(global_idx), "frame": int(f),
                    "instance": inst, "object_class": cls, "phase": verb, "destination": dest,
                    "position": float(pos), "subtask": text, "holdout": holdout,
                })
    return samples


def _diverse_samples(buffer, cfg, key: str, holdout: bool, rng) -> list[dict]:
    """One sample per chosen anchor of every (robot, class, phase) atom of a diverse buffer."""
    from lerobot.datasets.diverse_actor_selection import _atom_at, _atoms_by_episode

    p = cfg.probe_parameters
    atoms = _atoms_by_episode(buffer.selection.corpus, "subtask_atoms")
    cells: dict[tuple[str, str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row_index, row in enumerate(buffer.rows):
        robot = ROBOT_OF_SOURCE.get(row["source"])
        if row["corpus_key"] == "fmb" or robot is None:
            continue
        atom = _atom_at(atoms[(row["corpus_key"], str(row["episode_id"]))], int(row["anchor_frame"]))
        verb = str(atom["verb"]).lower()
        cls = _object_class(str(atom.get("object") or ""))
        if verb not in PHASES or cls is None:
            continue
        lo, hi = int(atom["start_timestep"]), int(atom["end_timestep_exclusive"])
        pos = (int(row["anchor_frame"]) - lo) / max(hi - lo, 1)
        cells[(robot, cls, verb)][str(row["episode_id"])].append(
            (float(row["anchor_s"]), row_index, pos, str(row["subtask"]), atom.get("container"), row["source"])
        )
    samples = []
    for (robot, cls, verb), by_ep in sorted(cells.items()):
        episodes = sorted(by_ep) if holdout else _cap_episodes(sorted(by_ep), p.conditions_episodes_per_cell, rng)
        for ep in episodes:
            rows = sorted(by_ep[ep])
            for k in _even_picks(len(rows), p.conditions_frames_per_episode_cell):
                anchor_s, row_index, pos, text, dest, source = rows[int(k)]
                samples.append({
                    "kind": "diverse", "source_key": key, "robot": robot, "source": source,
                    "episode": f"{source}/{ep}", "index": int(row_index), "frame": anchor_s,
                    "instance": cls, "object_class": cls, "phase": verb, "destination": dest,
                    "position": float(pos), "subtask": text, "holdout": holdout,
                })
    return samples


def _plan_summary(samples: list[dict]) -> dict:
    out: dict = defaultdict(lambda: defaultdict(lambda: {"frames": 0, "episodes": set()}))
    for s in samples:
        cell = out[f"{s['robot']}{' (holdout)' if s['holdout'] else ''}"][f"{s['object_class']}/{s['phase']}"]
        cell["frames"] += 1
        cell["episodes"].add(s["episode"])
    return {robot: {cell: {"frames": v["frames"], "episodes": len(v["episodes"])} for cell, v in cells.items()}
            for robot, cells in out.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Collection
# ──────────────────────────────────────────────────────────────────────────────

def _thumb_path(cache_dir: str, sample_index: int, camera: str) -> str:
    return os.path.join(cache_dir, "thumbs", f"{sample_index:04d}.{camera}.jpg")


def _save_thumbs(obs: dict, cache_dir: str, sample_index: int) -> None:
    """Both camera views of one sample, downsized, kept in the cache so ``mode=plot`` can
    show which frames the matrix was built from without a dataset or a second decode."""
    from PIL import Image

    for camera in THUMB_CAMERAS:
        tensor = obs.get(f"observation.images.{camera}")
        if tensor is None:
            continue
        # as_image only scales a [0,1] tensor; a [0,255] float comes back float.
        image = Image.fromarray(np.clip(as_image(tensor), 0, 255).astype(np.uint8))
        image.thumbnail((THUMB_PX, THUMB_PX), Image.Resampling.LANCZOS)
        image.save(_thumb_path(cache_dir, sample_index, camera), quality=85)


def _rebot_inputs(dataset, cfg, sample: dict, chunk_size: int) -> dict:
    """The deployment-regime frame without the gripper-event loss targets, which the
    capture never forwards and which not every ReBot root carries."""
    frame = probe_frame_inputs(dataset, cfg, sample["index"], chunk_size, with_gripper_event_targets=False)
    return {"obs": frame["obs"], "task": frame["task"], "subtask": frame["subtask"],
            "metadata": dict(DEPLOYMENT_METADATA),
            "extra": {"embodiment_index": sample["embodiment_index"]}}


@torch.no_grad()
def collect(adapter, cfg, samples: list[dict], datasets: dict, buffers: dict, cache_dir: str) -> None:
    """Every sample under both text conditions, one fp16 memmap per group."""
    makedirs(os.path.join(cache_dir, "thumbs"))
    n_rows = len(samples) * len(TEXT_CONDITIONS)
    seed = int(cfg.probe_parameters.random_seed)
    arrays: dict[str, np.memmap] = {}
    present = {g: np.zeros(n_rows, dtype=bool) for g in GROUP_NAMES}
    meta = []
    for i, s in enumerate(samples):
        if i % 50 == 0:
            logging.info(f"  [{i + 1}/{len(samples)}] {s['robot']} {s['episode']} @ {s['frame']} {s['object_class']}/{s['phase']}")
        inputs = (
            _rebot_inputs(datasets[s["source_key"]], cfg, s, adapter.chunk_size) if s["kind"] == "rebot"
            else _diverse_inputs(buffers[s["source_key"]], cfg, s)
        )
        _save_thumbs(inputs["obs"], cache_dir, i)
        for t, text in enumerate(TEXT_CONDITIONS):
            task, subtask = (inputs["task"], s["subtask"]) if text == "real" else (NEUTRAL_TASK, NEUTRAL_SUBTASK[s["phase"]])
            reps = adapter.capture_layer_representations(
                inputs["obs"], task, subtask=subtask, metadata=inputs["metadata"],
                extra_complementary=inputs["extra"], noise_seed=seed,
            )
            r = i * len(TEXT_CONDITIONS) + t
            for site, group in GROUPS:
                vec = reps[site].get(group)
                if vec is None:
                    continue
                if group not in arrays:
                    arrays[group] = np.lib.format.open_memmap(
                        os.path.join(cache_dir, f"{group}.npy"), mode="w+", dtype=np.float16, shape=(n_rows, *vec.shape)
                    )
                arrays[group][r] = vec.numpy()
                present[group][r] = True
            meta.append({**s, "row": r, "text": text, "task": task, "subtask_used": subtask})
    for group, arr in arrays.items():
        arr.flush()
        np.save(os.path.join(cache_dir, f"{group}.present.npy"), present[group])
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"rows": meta, "groups": {g: list(a.shape[1:]) for g, a in arrays.items()}}, f)


def _load_cache(cache_dir: str) -> tuple[list[dict], dict[str, np.memmap], dict[str, np.ndarray]]:
    meta = json.load(open(os.path.join(cache_dir, "meta.json")))
    arrays = {g: np.load(os.path.join(cache_dir, f"{g}.npy"), mmap_mode="r") for g in meta["groups"]}
    present = {g: np.load(os.path.join(cache_dir, f"{g}.present.npy")) for g in meta["groups"]}
    return meta["rows"], arrays, present


# ──────────────────────────────────────────────────────────────────────────────
# Matrices
# ──────────────────────────────────────────────────────────────────────────────

def _unit(z: np.ndarray) -> np.ndarray:
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def cell_matrix(z: np.ndarray, cell: np.ndarray, episode: np.ndarray, n_cells: int) -> np.ndarray:
    """$M[i,j]$ = mean inner product over pairs from different episodes. ``cell`` indexes
    the cell order (-1 = not a cell), ``episode`` is any hashable per frame. nan where a
    cell pair has no cross-episode frame pair."""
    d = z.shape[1]
    total = np.zeros((n_cells, d))
    count = np.zeros(n_cells)
    by_ep: list[dict] = [dict() for _ in range(n_cells)]
    for k in range(n_cells):
        idx = np.nonzero(cell == k)[0]
        if len(idx) == 0:
            continue
        total[k] = z[idx].sum(axis=0)
        count[k] = len(idx)
        for ep in np.unique(episode[idx]):
            sel = idx[episode[idx] == ep]
            by_ep[k][ep] = (z[sel].sum(axis=0), len(sel))
    m = np.full((n_cells, n_cells), np.nan)
    for i in range(n_cells):
        for j in range(i, n_cells):
            num = float(total[i] @ total[j])
            den = count[i] * count[j]
            for ep in by_ep[i].keys() & by_ep[j].keys():
                s_i, n_i = by_ep[i][ep]
                s_j, n_j = by_ep[j][ep]
                num -= float(s_i @ s_j)
                den -= n_i * n_j
            if den > 0:
                m[i, j] = m[j, i] = num / den
    return m


def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], 1)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(a[keep], b[keep]).correlation)


def _null_percentile(m_a: np.ndarray, m_b: np.ndarray, rho: float, rng) -> tuple[float, float]:
    """Permute B's cell labels: (95th percentile of the null rho, p-value of ``rho``)."""
    k = m_b.shape[0]
    perms = np.stack([rng.permutation(k) for _ in range(N_NULL)])
    permuted = m_b[perms[:, :, None], perms[:, None, :]]  # [P, K, K]
    iu = np.triu_indices(k, 1)
    ub = permuted[:, iu[0], iu[1]]  # [P, m]
    ua = _upper(m_a)
    keep = np.isfinite(ua) & np.all(np.isfinite(ub), axis=0)
    if keep.sum() < 3:
        return float("nan"), float("nan")
    ra = rankdata(ua[keep])
    rb = rankdata(ub[:, keep], axis=1)
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean(axis=1, keepdims=True)) / rb.std(axis=1, keepdims=True)
    null = (rb * ra).mean(axis=1)
    return float(np.percentile(null, 95)), float((null >= rho).mean())


def _split_half_reliability(z, cell, episode, n_cells, rng) -> float:
    episodes = np.unique(episode)
    if len(episodes) < 4:
        return float("nan")
    rhos = []
    for _ in range(N_SPLITS):
        half = set(rng.permutation(episodes)[: len(episodes) // 2].tolist())
        in_half = np.array([e in half for e in episode])
        m1 = cell_matrix(z[in_half], cell[in_half], episode[in_half], n_cells)
        m2 = cell_matrix(z[~in_half], cell[~in_half], episode[~in_half], n_cells)
        rhos.append(_spearman(_upper(m1), _upper(m2)))
    return float(np.nanmean(rhos))


def _organisation(m: np.ndarray, labels: list[tuple], instance_level: bool) -> dict[str, float]:
    """Standardised OLS weights of the upper triangle on same-class / same-phase
    (/ same-instance) indicators."""
    iu = np.triu_indices(m.shape[0], 1)
    y = m[iu]
    predictors = {
        "same_class": np.array([labels[i][-2] == labels[j][-2] for i, j in zip(*iu)], dtype=float),
        "same_phase": np.array([labels[i][-1] == labels[j][-1] for i, j in zip(*iu)], dtype=float),
    }
    if instance_level:
        predictors["same_instance"] = np.array([labels[i][0] == labels[j][0] for i, j in zip(*iu)], dtype=float)
    keep = np.isfinite(y)
    if keep.sum() < len(predictors) + 2:
        return {k: float("nan") for k in predictors}
    x = np.stack([v[keep] for v in predictors.values()], axis=1)
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-8)
    yy = (y[keep] - y[keep].mean()) / max(y[keep].std(), 1e-8)
    beta = np.linalg.lstsq(x, yy, rcond=None)[0]
    return {k: float(b) for k, b in zip(predictors, beta)}


# ──────────────────────────────────────────────────────────────────────────────
# Cross-robot decoding
# ──────────────────────────────────────────────────────────────────────────────

FACTORS = (("class", 1), ("phase", 2))  # name, position in the (instance, class, phase) cell tuple


def _shared_view(robot: dict, shared: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``robot``'s frames inside ``shared``: unit vectors centred on the mean of the
    shared-cell means, the cell index into ``shared`` and the episode of every frame."""
    index = {c: k for k, c in enumerate(shared)}
    cell = np.array([index.get(robot["cells"][c], -1) if c >= 0 else -1 for c in robot["cell"]])
    keep = cell >= 0
    raw, cell, episode = robot["raw"][keep], cell[keep], robot["episode"][keep]
    mu = np.stack([raw[cell == k].mean(axis=0) for k in range(len(shared))]).mean(axis=0)
    return _unit(raw - mu), cell, episode


def _factor_index(shared: list[tuple], position: int) -> tuple[list, np.ndarray]:
    """Distinct values of one factor over the shared cells, and each cell's index into them."""
    order = CLASSES if position == 1 else PHASES
    values = sorted({c[position] for c in shared}, key=order.index)
    return values, np.array([values.index(c[position]) for c in shared])


def _nearest(z: np.ndarray, means: np.ndarray) -> np.ndarray:
    return (z @ _unit(means).T).argmax(axis=1)


def _predict(z: np.ndarray, means: np.ndarray, shared: list[tuple]) -> dict[str, np.ndarray]:
    """Nearest-cosine decoding of the cell, and of each factor from the factor means."""
    preds = {"cell": _nearest(z, means)}
    for name, position in FACTORS:
        values, of_cell = _factor_index(shared, position)
        preds[name] = _nearest(z, np.stack([means[of_cell == f].mean(axis=0) for f in range(len(values))]))
    return preds


def _loeo_predictions(z: np.ndarray, cell: np.ndarray, episode: np.ndarray, shared: list[tuple]) -> dict[str, np.ndarray]:
    """Every frame decoded from its own robot's cell means with its episode left out."""
    onehot = np.eye(len(shared))[cell]
    total, count = onehot.T @ z, onehot.sum(axis=0)
    preds = {name: np.empty(len(z), dtype=int) for name in ("cell", *(n for n, _ in FACTORS))}
    for ep in np.unique(episode):
        sel = episode == ep
        means = (total - onehot[sel].T @ z[sel]) / (count - onehot[sel].sum(axis=0))[:, None]
        for name, p in _predict(z[sel], means, shared).items():
            preds[name][sel] = p
    return preds


def _balanced_accuracy(pred: np.ndarray, truth: np.ndarray, n: int) -> float:
    """Mean per-label recall; chance is 1 / n whatever the label counts."""
    return float(np.mean([(pred[truth == k] == k).mean() for k in range(n) if (truth == k).any()]))


def _accuracies(preds: dict, cell: np.ndarray, shared: list[tuple], prefix: str = "") -> dict[str, float]:
    out = {f"{prefix}cell": _balanced_accuracy(preds["cell"], cell, len(shared)), "cell_chance": 1 / len(shared)}
    for name, position in FACTORS:
        values, of_cell = _factor_index(shared, position)
        out[f"{prefix}{name}"] = _balanced_accuracy(preds[name], of_cell[cell], len(values))
        out[f"{name}_chance"] = 1 / len(values)
    return out


def _decode(src: dict, dst: dict, shared: list[tuple]) -> dict:
    """``dst``'s frames decoded with ``src``'s cell means (cross) and with their own
    leave-one-episode-out means (within), balanced accuracy for cell / class / phase."""
    z_s, cell_s, _ = _shared_view(src, shared)
    z_d, cell_d, ep_d = _shared_view(dst, shared)
    onehot = np.eye(len(shared))[cell_s]
    means_s = (onehot.T @ z_s) / onehot.sum(axis=0)[:, None]
    return {"n_cells": len(shared), "n_frames": len(z_d),
            **_accuracies(_predict(z_d, means_s, shared), cell_d, shared),
            **_accuracies(_loeo_predictions(z_d, cell_d, ep_d, shared), cell_d, shared, prefix="within_")}


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

def _robot_cells(rows: list[dict], robot: str, instance_level: bool) -> list[tuple]:
    """Cells (instance, class, phase) with >= MIN_EPISODES_PER_CELL training episodes."""
    episodes: dict[tuple, set] = defaultdict(set)
    for r in rows:
        if r["robot"] != robot or r["holdout"]:
            continue
        key = (r["instance"], r["object_class"], r["phase"]) if instance_level else (r["object_class"], r["object_class"], r["phase"])
        episodes[key].add(r["episode"])
    cells = [k for k, eps in episodes.items() if len(eps) >= MIN_EPISODES_PER_CELL]
    return sorted(cells, key=lambda k: (CLASSES.index(k[1]), k[0], PHASES.index(k[2])))


def _label(cell: tuple, instance_level: bool) -> str:
    return f"{cell[0]}/{cell[2]}" if instance_level else f"{cell[1]}/{cell[2]}"


def _cell_counts(cell: np.ndarray, episode: np.ndarray, n_cells: int) -> list[tuple[int, int]]:
    """(frames, episodes) per cell: the sample each matrix row rests on."""
    return [(int((cell == k).sum()), len(set(episode[cell == k]))) for k in range(n_cells)]


def _robot_totals(rows: list[dict], robot: str, cells: list[tuple]) -> tuple[int, int]:
    """Frames and episodes of ``robot`` inside the matrix's cells (training half)."""
    keep = [r for r in rows if r["robot"] == robot and not r["holdout"] and r["text"] == "real"
            and (r["object_class"], r["object_class"], r["phase"]) in set(cells)]
    return len(keep), len({r["episode"] for r in keep})


def analyze(rows: list[dict], arrays: dict, present: dict, cfg, output_dir: str) -> dict:
    seed = int(cfg.probe_parameters.random_seed)
    robots = [r for r in ROBOT_ORDER if any(x["robot"] == r and not x["holdout"] for x in rows)]
    class_cells = {r: _robot_cells(rows, r, instance_level=False) for r in robots}
    robots = [r for r in robots if len(class_cells[r]) >= MIN_SHARED_CELLS]
    rebot_instance_cells = _robot_cells(rows, REBOT, instance_level=True) if REBOT in robots else []
    pairs = [(a, b) for i, a in enumerate(robots) for b in robots[i + 1:]
             if len(set(class_cells[a]) & set(class_cells[b])) >= MIN_SHARED_CELLS]
    logging.info(f"  robots {robots}; pairs {pairs}")

    row_robot = np.array([r["robot"] for r in rows])
    row_holdout = np.array([r["holdout"] for r in rows])
    row_text = np.array([r["text"] for r in rows])
    row_episode = np.array([r["episode"] for r in rows])
    row_class_key = [(r["object_class"], r["object_class"], r["phase"]) for r in rows]
    row_inst_key = [(r["instance"], r["object_class"], r["phase"]) for r in rows]

    metrics, organisation, holdout_rows, decoding = [], [], [], []
    matrices: dict = {}  # (group, text, layer, robot, level) -> (M, labels, counts)
    for group in GROUP_NAMES:
        if group not in arrays:
            continue
        x = arrays[group]
        n_layers = x.shape[1]
        for text in TEXT_CONDITIONS:
            for layer in range(n_layers):
                rng = np.random.RandomState(seed + layer)
                z_all = np.asarray(x[:, layer, :], dtype=np.float32)
                per_robot: dict = {}
                for robot in robots:
                    train = (row_robot == robot) & (row_text == text) & ~row_holdout & present[group]
                    if train.sum() == 0:
                        continue
                    mu = z_all[train].mean(axis=0)
                    idx = np.nonzero(train)[0]
                    z = _unit(z_all[idx] - mu)
                    cells = class_cells[robot]
                    index = {c: k for k, c in enumerate(cells)}
                    cell = np.array([index.get(row_class_key[i], -1) for i in idx])
                    episode = row_episode[idx]
                    m = cell_matrix(z, cell, episode, len(cells))
                    rel = _split_half_reliability(z, cell, episode, len(cells), rng)
                    per_robot[robot] = {"m": m, "cells": cells, "rel": rel, "mu": mu, "z": z, "cell": cell,
                                        "raw": z_all[idx], "episode": episode}
                    matrices[(group, text, layer, robot, "class")] = (
                        m, [_label(c, False) for c in cells], _cell_counts(cell, episode, len(cells)))
                    organisation.append({"group": group, "text": text, "layer": layer, "robot": robot, "level": "class",
                                         "n_cells": len(cells), "reliability": rel, **_organisation(m, cells, False)})
                    if robot == REBOT and rebot_instance_cells:
                        index_i = {c: k for k, c in enumerate(rebot_instance_cells)}
                        cell_i = np.array([index_i.get(row_inst_key[i], -1) for i in idx])
                        m_i = cell_matrix(z, cell_i, episode, len(rebot_instance_cells))
                        matrices[(group, text, layer, robot, "instance")] = (
                            m_i, [_label(c, True) for c in rebot_instance_cells],
                            _cell_counts(cell_i, episode, len(rebot_instance_cells)))
                        organisation.append({"group": group, "text": text, "layer": layer, "robot": robot, "level": "instance",
                                             "n_cells": len(rebot_instance_cells), "reliability": float("nan"),
                                             **_organisation(m_i, rebot_instance_cells, True)})
                    # Held-out frames of this robot against the training cell means.
                    held = (row_robot == robot) & (row_text == text) & row_holdout & present[group]
                    if held.sum() > 0:
                        hidx = np.nonzero(held)[0]
                        zh = _unit(z_all[hidx] - mu)
                        means = np.stack([z[cell == k].mean(axis=0) for k in range(len(cells))])
                        scores = zh @ means.T
                        truth = np.array([index.get(row_class_key[i], -1) for i in hidx])
                        keep = truth >= 0
                        if keep.sum() > 0:
                            acc = float((scores[keep].argmax(axis=1) == truth[keep]).mean())
                            corr = float(np.nanmean([_spearman(scores[i], m[truth[i]]) for i in np.nonzero(keep)[0]]))
                            holdout_rows.append({"group": group, "text": text, "layer": layer, "robot": robot,
                                                 "n_frames": int(keep.sum()), "n_cells": len(cells),
                                                 "accuracy": acc, "chance": 1.0 / len(cells), "row_corr": corr})
                for a, b in pairs:
                    if a not in per_robot or b not in per_robot:
                        continue
                    shared = [c for c in per_robot[a]["cells"] if c in set(per_robot[b]["cells"])]
                    ia = [per_robot[a]["cells"].index(c) for c in shared]
                    ib = [per_robot[b]["cells"].index(c) for c in shared]
                    m_a = per_robot[a]["m"][np.ix_(ia, ia)]
                    m_b = per_robot[b]["m"][np.ix_(ib, ib)]
                    rho = _spearman(_upper(m_a), _upper(m_b))
                    null95, p_value = _null_percentile(m_a, m_b, rho, rng)
                    ceiling = float(np.sqrt(max(per_robot[a]["rel"], 0.0) * max(per_robot[b]["rel"], 0.0)))
                    metrics.append({
                        "group": group, "text": text, "layer": layer, "pair": f"{a}|{b}", "n_cells": len(shared),
                        "rho": rho, "null_p95": null95, "p_value": p_value,
                        "reliability_a": per_robot[a]["rel"], "reliability_b": per_robot[b]["rel"], "ceiling": ceiling,
                        "rho_over_ceiling": rho / ceiling if ceiling > 0.05 else float("nan"),
                        "null_p95_over_ceiling": null95 / ceiling if ceiling > 0.05 else float("nan"),
                    })
                    for src, dst in ((a, b), (b, a)):
                        decoding.append({"group": group, "text": text, "layer": layer, "src": src, "dst": dst,
                                         **_decode(per_robot[src], per_robot[dst], shared)})
            logging.info(f"  {group}/{text}: " + "  ".join(
                f"{r['pair']} {r['rho_over_ceiling']:.2f}" for r in metrics
                if r["group"] == group and r["text"] == text and r["layer"] == n_layers - 1))

    _write_csv(os.path.join(output_dir, "metrics.csv"), metrics)
    _write_csv(os.path.join(output_dir, "organisation.csv"), organisation)
    _write_csv(os.path.join(output_dir, "holdout.csv"), holdout_rows)
    _write_csv(os.path.join(output_dir, "decoding.csv"), decoding)

    peak = _peak_layers(metrics, pairs)
    plot_by_layer(metrics, pairs, os.path.join(output_dir, "sharing_by_layer.png"))
    headline_layer = peak.get("action", peak.get(next(iter(peak), "action"), 0))
    totals = {r: _robot_totals(rows, r, class_cells[r]) for r in robots}
    plot_matrices(matrices, robots, totals, headline_layer, os.path.join(output_dir, "matrices.png"))
    n_layers = max((k[2] for k in matrices), default=-1) + 1
    extra_layers = [int(l) for l in str(cfg.probe_parameters.conditions_layers or "").split(",") if l.strip()]
    extra_layers = [l for l in extra_layers if l != headline_layer and 0 <= l < n_layers]
    for layer in extra_layers:
        plot_matrices(matrices, robots, totals, layer, os.path.join(output_dir, f"matrices_L{layer}.png"))
    plot_decoding(decoding, pairs, [headline_layer, *extra_layers], os.path.join(output_dir, "decoding.png"))
    if rebot_instance_cells:
        plot_rebot_instances(matrices, totals, headline_layer, os.path.join(output_dir, "rebot_instances.png"))
    cache_dir = os.path.join(output_dir, "cache")
    frames_written = [r for r in robots
                      if plot_cell_frames(rows, class_cells[r], cache_dir, r, os.path.join(output_dir, f"frames_{r}.png"))]
    if not frames_written:
        logging.warning("  no cache/thumbs: frames_<robot>.png skipped (re-run mode=collect to write them)")

    summary = _summary(rows, metrics, organisation, holdout_rows, decoding, pairs, robots, class_cells, peak, headline_layer)
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _write_manifest(output_dir, summary, pairs, frames_written, extra_layers)
    return summary


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fields: dict[str, None] = {}
    for row in rows:
        fields.update(dict.fromkeys(row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), restval="")
        writer.writeheader()
        writer.writerows(rows)


def _rebot_pairs(pairs) -> list[tuple[str, str]]:
    return [p for p in pairs if REBOT in p] or list(pairs)


def _peak_layers(metrics: list[dict], pairs) -> dict[str, int]:
    """Per group: the layer where the mean rho/ceiling over the rebot pairs (real text) peaks."""
    peak = {}
    focus = {f"{a}|{b}" for a, b in _rebot_pairs(pairs)}
    for group in GROUP_NAMES:
        by_layer: dict[int, list] = defaultdict(list)
        for r in metrics:
            if r["group"] == group and r["text"] == "real" and r["pair"] in focus and np.isfinite(r["rho_over_ceiling"]):
                by_layer[r["layer"]].append(r["rho_over_ceiling"])
        if by_layer:
            peak[group] = max(by_layer, key=lambda l: float(np.mean(by_layer[l])))
    return peak


def _summary(rows, metrics, organisation, holdout_rows, decoding, pairs, robots, class_cells, peak, headline_layer) -> dict:
    def at(group, text, layer):
        return {r["pair"]: r for r in metrics if r["group"] == group and r["text"] == text and r["layer"] == layer}

    n_layers = max((r["layer"] for r in metrics), default=-1) + 1
    summary = {
        "n_samples": len({r["row"] // len(TEXT_CONDITIONS) for r in rows}),
        "n_rows": len(rows),
        "robots": robots,
        "pairs": [f"{a}|{b}" for a, b in pairs],
        "cells": {r: [f"{c[1]}/{c[2]}" for c in class_cells[r]] for r in robots},
        "plan": _plan_summary([r for r in rows if r["text"] == "real"]),
        "n_layers": n_layers,
        "peak_layer": peak,
        "headline_layer": headline_layer,
        "headline": {},
    }
    for group in GROUP_NAMES:
        layer = peak.get(group)
        if layer is None:
            continue
        entry = {"layer": layer}
        for text in TEXT_CONDITIONS:
            entry[text] = {pair: {"rho_over_ceiling": r["rho_over_ceiling"], "rho": r["rho"], "ceiling": r["ceiling"],
                                  "null_p95": r["null_p95"], "p_value": r["p_value"], "n_cells": r["n_cells"]}
                           for pair, r in at(group, text, layer).items()}
        entry["organisation"] = {f"{o['robot']}/{o['level']}": {k: o[k] for k in ("same_class", "same_phase", "same_instance") if k in o}
                                 for o in organisation if o["group"] == group and o["text"] == "real" and o["layer"] == layer}
        entry["holdout"] = {h["robot"]: {"accuracy": h["accuracy"], "chance": h["chance"], "row_corr": h["row_corr"], "n_frames": h["n_frames"]}
                            for h in holdout_rows if h["group"] == group and h["text"] == "real" and h["layer"] == layer}
        summary[f"peak_{group}"] = entry
    # Flat headline numbers for the manifest: the action group at its peak layer.
    for text in TEXT_CONDITIONS:
        for pair, r in at("action", text, headline_layer).items():
            summary["headline"][f"{text}.{pair.replace('|', '_')}"] = r["rho_over_ceiling"]
    for h in holdout_rows:
        if h["group"] == "action" and h["text"] == "real" and h["layer"] == headline_layer:
            summary["headline"][f"holdout_accuracy.{h['robot']}"] = h["accuracy"]
            summary["headline"][f"holdout_chance.{h['robot']}"] = h["chance"]
    # Cross-robot decoding at the headline layer, action group; corpus -> ReBot as headline numbers.
    summary["decoding"] = {}
    for d in decoding:
        if d["group"] == "action" and d["layer"] == headline_layer:
            summary["decoding"][f"{d['text']}.{d['src']}_to_{d['dst']}"] = {k: v for k, v in d.items() if k not in ("group", "layer")}
            if d["dst"] == REBOT:
                for name, _ in FACTORS:
                    summary["headline"][f"decoding.{d['text']}.{d['src']}_to_rebot.{name}"] = d[name]
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

GROUP_COLORS = {"img_wrist_0": "#2ca02c", "img_external_0": "#d62728", "subtask": "#ff7f0e",
                "action_output": "#9467bd", "action": "#1f77b4"}


def plot_by_layer(metrics: list[dict], pairs, path: str) -> None:
    """rho / ceiling against layer: one row per robot pair, one column per text condition,
    one curve per token group; dotted = the largest null 95th percentile over the groups."""
    pair_names = [f"{a}|{b}" for a, b in sorted(pairs, key=lambda p: (REBOT not in p, p))]
    if not pair_names:
        return
    fig, axes = plt.subplots(len(pair_names), len(TEXT_CONDITIONS), figsize=(5.2 * len(TEXT_CONDITIONS), 2.6 * len(pair_names)),
                             squeeze=False, sharex=True, sharey=True)
    for i, pair in enumerate(pair_names):
        for j, text in enumerate(TEXT_CONDITIONS):
            ax = axes[i][j]
            null_by_layer: dict[int, list] = defaultdict(list)
            for group in GROUP_NAMES:
                series = sorted((r["layer"], r["rho_over_ceiling"]) for r in metrics
                                if r["pair"] == pair and r["text"] == text and r["group"] == group)
                if not series:
                    continue
                ax.plot([l for l, _ in series], [v for _, v in series], marker="o", markersize=2.5, linewidth=1.3,
                        color=GROUP_COLORS[group], linestyle="--" if group == "action" else "-", label=group)
                for r in metrics:
                    if r["pair"] == pair and r["text"] == text and r["group"] == group:
                        null_by_layer[r["layer"]].append(r["null_p95_over_ceiling"])
            if null_by_layer:
                layers = sorted(null_by_layer)
                ax.plot(layers, [np.nanmax(null_by_layer[l]) for l in layers], color="#888", linestyle=":", linewidth=1.2,
                        label="null 95th pct")
            ax.axhline(1.0, color="#bbb", linewidth=0.8)
            ax.axhline(0.0, color="#bbb", linewidth=0.8)
            ax.set_title(f"{pair.replace('|', ' vs ')}  ·  {text} text", fontsize=9)
            ax.grid(alpha=0.3)
            if i == len(pair_names) - 1:
                ax.set_xlabel("layer")
            if j == 0:
                ax.set_ylabel("rho / ceiling")
    finite = [r["rho_over_ceiling"] for r in metrics if np.isfinite(r["rho_over_ceiling"])]
    top = max(1.25, min(2.5, max(finite) + 0.1)) if finite else 1.25
    axes[0][0].set_ylim(-0.5, top)
    axes[0][0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax, m: np.ndarray, labels: list[str], counts: list[tuple[int, int]], title: str) -> None:
    """Row labels carry the frames / episodes the row rests on; columns stay bare."""
    finite = m[np.isfinite(m)]
    vmax = float(np.abs(finite).max()) if finite.size else 1.0
    ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels([f"{l}  {nf}f/{ne}e" for l, (nf, ne) in zip(labels, counts)], fontsize=6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(m[i, j]) > 0.6 * vmax else "black")
    ax.set_title(title, fontsize=8)


def plot_matrices(matrices: dict, robots: list[str], totals: dict, layer: int, path: str) -> None:
    """Class-level matrices of every robot at ``layer``, real text: rows action / wrist.
    ``totals`` = (frames, episodes) per robot for the panel titles."""
    groups = [g for g in ("action", "img_wrist_0") if any(k[0] == g for k in matrices)]
    if not groups or not robots:
        return
    fig, axes = plt.subplots(len(groups), len(robots), figsize=(3.4 * len(robots), 3.4 * len(groups)), squeeze=False)
    for i, group in enumerate(groups):
        for j, robot in enumerate(robots):
            entry = matrices.get((group, "real", layer, robot, "class"))
            ax = axes[i][j]
            if entry is None:
                ax.axis("off")
                continue
            nf, ne = totals[robot]
            _heatmap(ax, entry[0], entry[1], entry[2], f"{robot} · {group} · L{layer}\n{nf} frames / {ne} episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_rebot_instances(matrices: dict, totals: dict, layer: int, path: str) -> None:
    groups = [g for g in ("action", "img_wrist_0") if (g, "real", layer, REBOT, "instance") in matrices]
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 5.5), squeeze=False)
    nf, ne = totals[REBOT]
    for j, group in enumerate(groups):
        m, labels, counts = matrices[(group, "real", layer, REBOT, "instance")]
        _heatmap(axes[0][j], m, labels, counts, f"rebot instances · {group} · L{layer}\n{nf} frames / {ne} episodes")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _episode_short(episode: str) -> str:
    """'droid/droid__AUTOLab__ep000401' -> 'AUTOLab__ep000401'; 'rebot_all-annotated-v1/12' -> 'ep 12'."""
    source, _, ep = episode.rpartition("/")
    if ep.startswith(f"{source}__"):
        return ep[len(source) + 2:]
    return f"ep {ep}" if ep.isdigit() else ep


def plot_cell_frames(rows: list[dict], cells: list[tuple], cache_dir: str, robot: str, path: str) -> bool:
    """The frames behind one robot's class-level matrix. Per cell, N_SHOW different
    episodes: the first window frame of one and the last of another, so the span the
    diagonal averages over is visible; external view over wrist view, captioned with
    the episode, the position in the window and the real subtask text. False when the
    cache carries no thumbnails."""
    from PIL import Image

    if not os.path.isdir(os.path.join(cache_dir, "thumbs")):
        return False
    by_cell: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["robot"] == robot and not r["holdout"] and r["text"] == "real":
            by_cell[(r["object_class"], r["object_class"], r["phase"])][r["episode"]].append(r)
    classes = sorted({c[1] for c in cells}, key=CLASSES.index)
    n_frames, n_episodes = _robot_totals(rows, robot, cells)
    n_rows, n_cols = 2 * len(classes), N_SHOW * len(PHASES)
    # Tile height follows this robot's camera aspect, plus the two caption lines.
    first = next(r for c in cells for eps in by_cell[c].values() for r in eps)
    with Image.open(_thumb_path(cache_dir, first["row"] // len(TEXT_CONDITIONS), THUMB_CAMERAS[0])) as im:
        aspect = im.height / im.width
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, (1.9 * aspect + 0.32) * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, cls in enumerate(classes):
        for c, camera in enumerate(THUMB_CAMERAS):
            axes[2 * i + c][0].text(-0.04, 0.5, f"{cls}\n{camera}", transform=axes[2 * i + c][0].transAxes,
                                    ha="right", va="center", fontsize=7, rotation=90)
        for j, phase in enumerate(PHASES):
            cell = (cls, cls, phase)
            episodes = sorted(by_cell[cell]) if cell in cells else []
            for k, e in enumerate(_even_picks(len(episodes), N_SHOW)):
                frames = sorted(by_cell[cell][episodes[int(e)]], key=lambda r: r["position"])
                r = frames[0] if k == 0 else frames[-1]
                for c, camera in enumerate(THUMB_CAMERAS):
                    ax = axes[2 * i + c][N_SHOW * j + k]
                    thumb = _thumb_path(cache_dir, r["row"] // len(TEXT_CONDITIONS), camera)
                    if os.path.isfile(thumb):
                        ax.imshow(Image.open(thumb))
                    else:
                        ax.text(0.5, 0.5, f"no {camera}", transform=ax.transAxes, ha="center", va="center", fontsize=6)
                    if c == 0:
                        ax.set_title(f"{cls}/{phase} · {_episode_short(r['episode'])} @{r['position']:.1f}\n"
                                     f"\"{r['subtask']}\"", fontsize=5)
    fig.suptitle(f"{robot} · {n_frames} frames / {n_episodes} episodes in the matrix · per cell {N_SHOW} of its "
                 f"episodes shown, first window frame of one and last of the other", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


FACTOR_COLORS = {"class": "#1f77b4", "phase": "#ff7f0e"}


def plot_decoding(decoding: list[dict], pairs, layers: list[int], path: str) -> None:
    """ReBot pairs, both directions: class and phase balanced accuracy of ``dst``'s frames
    decoded with ``src``'s cell means (bars), the same frames with their own robot's
    leave-one-episode-out means (black tick) and chance (dotted). Rows: group x text;
    columns: layers."""
    directions = [(s, d) for a, b in _rebot_pairs(pairs) for s, d in ((a, b), (b, a))]
    groups = [g for g in ("action", "img_wrist_0") if any(r["group"] == g for r in decoding)]
    if not directions or not groups:
        return
    panels = [(g, t) for g in groups for t in TEXT_CONDITIONS]
    fig, axes = plt.subplots(len(panels), len(layers), figsize=(3.6 * len(layers), 2.7 * len(panels)),
                             squeeze=False, sharey=True)
    width, x = 0.36, np.arange(len(directions))
    for i, (group, text) in enumerate(panels):
        for j, layer in enumerate(layers):
            ax = axes[i][j]
            at = {(r["src"], r["dst"]): r for r in decoding if r["group"] == group and r["text"] == text and r["layer"] == layer}
            for k, (name, _) in enumerate(FACTORS):
                xs = x + (k - 0.5) * width
                get = lambda key: [at[d][key] if d in at else np.nan for d in directions]  # noqa: E731
                ax.bar(xs, get(name), width, color=FACTOR_COLORS[name], label=f"{name}, cross-robot")
                ax.hlines(get(f"within_{name}"), xs - width / 2, xs + width / 2, color="black", linewidth=1.5,
                          label="within-robot, leave-one-episode-out" if k == 0 else None)
                ax.hlines(get(f"{name}_chance"), xs - width / 2, xs + width / 2, color=FACTOR_COLORS[name],
                          linestyle=":", linewidth=1.2, label="chance" if k == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{s}\u2192{d}" for s, d in directions], rotation=45, ha="right", fontsize=6)
            ax.set_title(f"{group} · {text} text · L{layer}", fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
            if j == 0:
                ax.set_ylabel("balanced accuracy", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Manifest + pipeline
# ──────────────────────────────────────────────────────────────────────────────

def _write_manifest(output_dir: str, summary: dict, pairs, frames_written: list[str], extra_layers: list[int]) -> dict:
    layer = summary["headline_layer"]
    metrics = []
    for a, b in sorted(pairs, key=lambda p: (REBOT not in p, p)):
        key = f"real.{a}_{b}"
        if key in summary["headline"]:
            metrics.append(Metric(f"headline.{key}", f"{a} vs {b}: rho / ceiling, action tokens, layer {layer}, real text",
                                  good="high", fmt=2, primary=REBOT in (a, b)))
        key = f"neutral.{a}_{b}"
        if key in summary["headline"]:
            metrics.append(Metric(f"headline.{key}", f"{a} vs {b}: rho / ceiling, action tokens, layer {layer}, neutral text",
                                  good="high", fmt=2))
    for robot in summary["robots"]:
        key = f"holdout_accuracy.{robot}"
        if key in summary["headline"]:
            metrics.append(Metric(f"headline.{key}", f"{robot} held-out frames: nearest training cell, action tokens, layer {layer}",
                                  good="high", fmt=2, note=f"chance {summary['headline'].get(f'holdout_chance.{robot}', float('nan')):.2f}"))
    for key, entry in summary["decoding"].items():
        text, direction = key.split(".", 1)
        if entry["dst"] != REBOT:
            continue
        for name, _ in FACTORS:
            if text == "neutral" and name == "phase":
                continue
            metrics.append(Metric(
                f"headline.decoding.{key}.{name}",
                f"ReBot frames decoded with {entry['src']}'s cell means: {name}, action tokens, layer {layer}, {text} text",
                good="high", fmt=2, primary=(text == "neutral"),
                note=f"within-ReBot leave-one-episode-out {entry[f'within_{name}']:.2f}, chance {entry[f'{name}_chance']:.2f}"))
    metrics.append(Metric("n_samples", "Frames captured (x2 text conditions)", good="none", fmt=0))
    panels = [
        Panel("sharing_by_layer.png", "rho / ceiling against depth, one row per robot pair, real and neutral text",
              how="1 = the two robots arrange the shared object x phase cells identically, up to what each robot's own "
                  "split-half reliability allows; 0 = unrelated; dotted = 95th percentile of the label-permutation null. "
                  "Left column has the object words in the prompt, right column has them removed: structure that stays "
                  "on the right came through the images.", primary=True),
        Panel("matrices.png", "Each robot's class-level similarity matrix at the headline layer (action and wrist groups)",
              how="Cells are object/phase; entries are mean cosine between frames of different episodes, after per-robot "
                  "centering. Read the block structure: same-object blocks vs same-phase stripes. Two robots share a "
                  "representation when their block patterns match, whatever the absolute values.", primary=True),
        Panel("rebot_instances.png", "ReBot's instance-level matrix at the headline layer",
              how="Spray bottle and pill bottle nearer each other than either to the cup = bottle is a category, not two "
                  "memorised objects; sock vs shirt likewise for cloth."),
    ]
    panels.append(Panel("decoding.png", "Cross-robot decoding: ReBot pairs, both directions, class and phase",
                        how="src\u2192dst = dst's frames decoded by nearest cosine to src's cell means over the cells both "
                            "robots have. Bars: cross-robot balanced accuracy; black tick: the same frames decoded with "
                            "their own robot's leave-one-episode-out means (the ceiling); dotted: chance. Bars at the "
                            "tick = the same directions carry the code on both robots; bars at chance under a high "
                            "tick = separate codes with the same shape, which the matrices alone cannot tell apart.",
                        primary=True))
    for l in extra_layers:
        panels.append(Panel(f"matrices_L{l}.png", f"The same class-level matrices at layer {l}",
                            how="Same reading as the headline figure; compare block patterns across depth."))
    for robot in frames_written:
        panels.append(Panel(f"frames_{robot}.png", f"{robot}: the frames behind its matrix, one tile per cell",
                            how="Rows are object class x camera, columns are phase x two different episodes: the first "
                                "window frame of one episode and the last of another (@x = position in the subtask "
                                "window, 0 start, 1 end). The diagonal of the matrix is the mean cosine between frames "
                                "like these two; the caption is the subtask text the model was given under the "
                                "real-text condition.", primary=True))
    return write_index(
        output_dir, sys.modules[__name__], title="Conditions matrix", group="Representation",
        claim="Are object and phase representations shared across robots: same relational geometry over the cells, "
              "with and without the object words in the prompt?",
        summary=summary, metrics=metrics, panels=panels, status="info",
        extra={"peak_layer": summary["peak_layer"], "cells": summary["cells"]},
        see_also=["domain_representations", "input_swap", "subtask_sweep"],
    )


def _open_diverse_buffers(cfg) -> dict:
    """Training buffer (cache when present, video otherwise) and the held-out buffer (video)."""
    from lerobot.datasets.diverse_actor_selection import (
        holdout_actor_selection, open_federated_corpus, select_actor_anchors,
    )
    from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer
    from lerobot.rl.data_sources.diverse_actor_cache import resolve_cache
    from lerobot.rl.data_sources.diverse_integration import sample_spec_from_config

    diverse_cfg = cfg.diverse
    corpus = open_federated_corpus(diverse_cfg.root)
    spec = sample_spec_from_config(cfg)
    selection = select_actor_anchors(corpus)
    cache = resolve_cache(
        diverse_cfg.root, diverse_cfg.cache_dir or getattr(cfg, "buffer_cache_dir", None),
        spec, selection, cache_policy="fallback",
    )
    buffers = {
        "diverse": DiverseActorBuffer(selection, spec, cache=cache, render_automatic_quality=diverse_cfg.render_automatic_quality),
    }
    holdout = holdout_actor_selection(corpus)
    if holdout.rows:
        buffers["diverse_holdout"] = DiverseActorBuffer(
            holdout, spec, cache=None, render_automatic_quality=diverse_cfg.render_automatic_quality,
        )
    return buffers


def _rebot_roots(cfg) -> list[str]:
    p = cfg.probe_parameters
    if p.conditions_rebot_roots:
        return [r.strip() for r in str(p.conditions_rebot_roots).split(",") if r.strip()]
    return [str(s.root) for s in (getattr(cfg.dataset, "sources", None) or []) if getattr(s, "root", None)]


def _has_windows(root) -> bool:
    return os.path.isfile(os.path.join(str(root), "meta", "subtask_windows.json"))


def run(adapter, dataset, cfg, output_dir: str) -> dict | None:
    """``dataset`` is the held-out ReBot set (its windows are scored, never fitted); the
    ReBot training roots come from ``conditions_rebot_roots`` or the config's sources; the
    diverse side from ``cfg.diverse`` plus ``<root>/holdout_episodes.json``."""
    p = cfg.probe_parameters
    makedirs(output_dir)
    cache_dir = os.path.join(output_dir, "cache")
    if p.mode in ("collect", "all"):
        rng = np.random.RandomState(int(p.random_seed))
        datasets: dict = {}
        samples: list[dict] = []
        for root in _rebot_roots(cfg):
            if not _has_windows(root):
                logging.warning(f"  {root}: no meta/subtask_windows.json, skipped")
                continue
            name = os.path.basename(os.path.normpath(root))
            datasets[name] = load_extra_dataset(cfg.dataset.repo_id, root)
            samples += _rebot_samples(datasets[name], cfg, name, holdout=False, rng=rng)
        if dataset is not None and _has_windows(dataset.root):
            name = f"holdout:{os.path.basename(os.path.normpath(str(dataset.root)))}"
            datasets[name] = dataset
            samples += _rebot_samples(dataset, cfg, name, holdout=True, rng=rng)
        buffers: dict = {}
        if getattr(cfg, "diverse", None) is not None and cfg.diverse.enabled:
            buffers = _open_diverse_buffers(cfg)
            samples += _diverse_samples(buffers["diverse"], cfg, "diverse", holdout=False, rng=rng)
            if "diverse_holdout" in buffers:
                samples += _diverse_samples(buffers["diverse_holdout"], cfg, "diverse_holdout", holdout=True, rng=rng)
        # Which frames are collected is decided above; this only fixes the order they are
        # read in. Cell-major order walks every episode once per cell, and each hop is a
        # fresh video seek — storage order keeps the decoder moving forward.
        samples.sort(key=lambda s: (s["kind"], s["source_key"], s["index"]))
        plan = _plan_summary(samples)
        for robot, cells in plan.items():
            logging.info(f"  {robot}: " + "  ".join(f"{c} {v['frames']}f/{v['episodes']}e" for c, v in cells.items()))
        logging.info(f"  {len(samples)} frames x {len(TEXT_CONDITIONS)} text conditions")
        collect(adapter, cfg, samples, datasets, buffers, cache_dir)
    if p.mode in ("plot", "all"):
        rows, arrays, present = _load_cache(cache_dir)
        return analyze(rows, arrays, present, cfg, output_dir)
    return None


@parser.wrap()
def cli(cfg: ProbeConditionsMatrixConfig):
    init_logging()
    device = get_safe_torch_device(try_device=cfg.policy.device)
    dataset = None
    val_path = getattr(cfg, "val_dataset_path", None)
    if val_path:
        dataset = load_extra_dataset(cfg.dataset.repo_id, val_path)
    adapter = None
    if cfg.probe_parameters.mode in ("collect", "all"):
        adapter = ProbablePolicy.for_config(cfg, device, dataset=dataset)
    step = 0
    for part in str(getattr(cfg.policy, "pretrained_path", "") or "").split(os.sep):
        if part.isdigit():
            step = int(part)
    run(adapter, dataset, cfg, os.path.join(
        cfg.probe_parameters.output_dir, "validation", f"step_{step:08d}", "conditions_matrix"))


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
