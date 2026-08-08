"""
Generic, policy-agnostic helpers for probe scripts.

Covers dataset frame reading, episode sampling, PCA/UMAP fitting, and matplotlib
/ Plotly style helpers used across probes. Anything that depends on a specific
policy lives in the per-policy adapter (``probes.adapters.<policy>``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import textwrap
import warnings
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.action_metrics import TRAJECTORY_ERROR_KEYS, trajectory_error_components


# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

EP_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff",
    "#9a6324", "#800000", "#aaffc3", "#808000", "#000075",
    "#a9a9a9", "#ffd8b1", "#fabed4", "#fffac8", "#e0e0e0",
]

SEQ_CMAPS = ["Reds",   "Blues",   "Greens",  "Oranges", "Purples",
             "copper", "cool",    "spring",  "winter",  "autumn"]

DS_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
             "#42d4f4", "#f032e6", "#bfef45", "#469990"]


# The target embodiment: rebot B601, 7-DOF. Anything else falls back to indices.
REBOT_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll", "gripper",
]


def joint_names_for_dim(action_dim: int) -> list[str]:
    if action_dim == len(REBOT_JOINT_NAMES):
        return REBOT_JOINT_NAMES
    return [f"joint_{i}" for i in range(action_dim)]


def as_image(tensor) -> "np.ndarray":
    """A camera observation as an ``imshow``-able HWC uint8 array."""
    image = tensor.detach().float().cpu().squeeze()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = image.numpy()
    if array.max() <= 1.0:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    return array


# ──────────────────────────────────────────────────────────────────────────────
# Filesystem
# ──────────────────────────────────────────────────────────────────────────────

def makedirs(*paths: str) -> None:
    """Create each path (and parents) if missing."""
    for p in paths:
        os.makedirs(p, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Deployment regime
# ──────────────────────────────────────────────────────────────────────────────

# The metadata clause a rollout asks for: the best behaviour the steering offers.
# Probes that omit the clause entirely land in a prompt regime that covers ~1.35%
# of training samples (subtask 0.3 x metadata 0.15 x history 0.3) and is not the
# deployment regime either. `metadata_steering` is the probe that varies this.
DEPLOYMENT_METADATA = {"quality": 5, "mistake": False}

_PACK_DROPOUT_FIELDS = (
    "subtask_dropout",
    "metadata_dropout",
    "history_dropout",
    "rgb_dropout",
)


@contextlib.contextmanager
def suppress_pack_dropout(preprocessor):
    """Zero the pack step's training-time dropouts for the duration of a probe forward.

    The MolmoAct2 pack step fires subtask/metadata/history/RGB dropout
    whenever the transition carries an ACTION (`build_action_labels`,
    processor_molmoact2.py), which every attention/Jacobian capture does because it
    drives the flow loss. Left on, each probe frame independently loses the wrist
    camera (rgb_dropout 0.15) or the whole short-term block (history_dropout 0.3)
    from an unseeded `random.random()` draw — the overlay videos flicker and the
    episode-wide p98 vmax they are normalized against is computed over a mixture.

    None of these dropouts zeroes a tensor; each builds a mask (history_on, the
    camera's <im_patch> attention span) or drops a prompt clause. Holding the
    *probability* at zero is therefore the right lever: `random.random() < 0.0` is
    never true, so the mask is never constructed. It also leaves the config
    (`pointmap_config.rgb_dropout_prob` etc.) authoritative and untouched. The
    model-side modality dropout in DepthPointmapEncoder is gated on `self.training`
    instead, so `policy.eval()` already covers it.

    THREADING: this mutates the shared pack step in place — probes are handed the
    same preprocessor instance the training loop packs with. That is safe only
    because validation runs synchronously in the training thread (async_prefetch
    backgrounds `ReplayBuffer.sample`, never the pack step), the same contract the
    trainer's own `step.prompt_mode` swap relies on. Do not open this context from a
    thread that can run alongside training batch packing.
    """
    saved: list[tuple[Any, str, float]] = []
    for step in getattr(preprocessor, "steps", []) or []:
        for name in _PACK_DROPOUT_FIELDS:
            value = getattr(step, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value:
                saved.append((step, name, value))
                setattr(step, name, 0.0)
    try:
        yield
    finally:
        for step, name, value in saved:
            setattr(step, name, value)


def probe_image_stride(cfg) -> int:
    """Frames between stored image/depth rows (`policy.image_stride`), floored at 1."""
    return max(int(getattr(cfg.policy, "image_stride", 1) or 1), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_probe_dataset(cfg) -> LeRobotDataset:
    """Load the dataset a standalone probe CLI should run on.

    ``make_dataset`` resolves ``cfg.dataset.repo_id`` + ``cfg.dataset.root``, but an
    offline-RL config declares its data under ``dataset.sources`` and leaves ``root``
    unset — so the plain call goes looking on the Hub for a repo_id that only names the
    collection. Fall back to the normalization source's root (the same one rl_offline
    trains against and hands the probes as their reference dataset).
    """
    from lerobot.datasets.factory import make_dataset

    if cfg.dataset.root is None:
        from lerobot.rl.offline_dataset_utils import get_offline_dataset_sources

        sources = get_offline_dataset_sources(cfg)
        if sources and sources[0].root is not None:
            cfg.dataset.root = sources[0].root
            logging.info(f"[probe] dataset.root unset; using sources[0]: {cfg.dataset.root}")

    dataset = make_dataset(cfg)
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Dataset frame reading (no policy involvement)
# ──────────────────────────────────────────────────────────────────────────────

def build_episode_index(dataset) -> dict[int, list[int]]:
    """Map episode_index → sorted list of global frame indices."""
    ep_to_indices: dict[int, list[int]] = {}
    for global_idx in range(len(dataset)):
        ep_idx = dataset.hf_dataset[global_idx]["episode_index"].item()
        ep_to_indices.setdefault(ep_idx, []).append(global_idx)
    for ep_idx in ep_to_indices:
        ep_to_indices[ep_idx].sort()
    return ep_to_indices


def load_extra_dataset(repo_id: str, root: str) -> LeRobotDataset:
    """Load an additional LeRobot dataset from a local *root* directory."""
    ds = LeRobotDataset(repo_id=repo_id, root=root)
    ds.delta_timestamps = None
    ds.delta_indices = None
    return ds


def dataset_display_name(dataset, fallback_root: str | os.PathLike | None = None) -> str:
    """Return a stable short name for a dataset object or fallback root."""
    root = getattr(dataset, "root", None) or fallback_root
    if root is None:
        return "dataset"
    return os.path.basename(os.path.normpath(os.fspath(root)))


_SUBTASK_INDEX_CACHE: dict[str, "torch.Tensor | None"] = {}


def _dataset_subtask_indices(dataset):
    """Per-frame subtask indices for a dataset, or ``None``, cached by root.

    These datasets carry no frame-level ``subtask_index`` column: the reviewed
    labels live in ``meta/subtask_windows.json`` and the offline collection
    materializes them at load time (`offline_dataset_utils._subtask_indices_from_windows`).
    Probes read the dataset directly, so without this every probe prompt silently
    loses its subtask clause — which is most of what the clause-level diagnostics
    are trying to measure.
    """
    key = str(getattr(dataset, "root", id(dataset)))
    if key not in _SUBTASK_INDEX_CACHE:
        from lerobot.rl.offline_dataset_utils import _subtask_indices_from_windows

        try:
            _SUBTASK_INDEX_CACHE[key] = _subtask_indices_from_windows(dataset, len(dataset))
        except (ValueError, KeyError, OSError) as exc:
            logging.warning(f"[probe] could not read meta/subtask_windows.json for {key}: {exc}")
            _SUBTASK_INDEX_CACHE[key] = None
    return _SUBTASK_INDEX_CACHE[key]


def get_subtask_idx(dataset, global_idx: int) -> int:
    """Read the subtask index for a dataset frame; returns -1 if unavailable."""
    frame_row = dataset.hf_dataset[global_idx]
    for key in ("subtask_index", "complementary_info.subtask_index"):
        if key in frame_row:
            val = frame_row[key]
            return val.item() if isinstance(val, torch.Tensor) else int(val)

    indices = _dataset_subtask_indices(dataset)
    if indices is not None and 0 <= global_idx < len(indices):
        return int(indices[global_idx])
    return -1


def get_subtask_str(dataset, subtask_idx: int) -> str:
    """Look up a subtask description string by index; returns "" if unavailable."""
    if subtask_idx < 0:
        return ""
    meta = getattr(dataset, "meta", None)
    subtasks_df = getattr(meta, "subtasks", None) if meta is not None else None
    if subtasks_df is None:
        return ""
    try:
        if hasattr(subtasks_df, "columns") and "subtask_index" in subtasks_df.columns:
            rows = subtasks_df[subtasks_df["subtask_index"] == subtask_idx]
            if not rows.empty:
                return str(rows.iloc[0].name)
        if subtask_idx in subtasks_df.index:
            return str(subtasks_df.loc[subtask_idx, "subtask"])
    except Exception:
        return ""
    return ""


def subtask_group(subtask: str) -> str:
    """Collapse an object-specific subtask to its verb.

    "release the red shirt in the bin" → "release", so the by-subtask plots show a
    handful of phases instead of one colour per object.
    """
    return subtask.split()[0] if subtask else "?"


def frame_metadata_lookup(dataset) -> dict[int, dict]:
    """global frame index → ``{"quality": int, "mistake": bool}``.

    Same spans training uses (`ReplayBuffer.materialize_metadata`): quality
    broadcasts over the episode, mistake over its 4 s window. Returns ``{}`` when
    the dataset has not been through `metadata_annotate.py`.
    """
    from lerobot.rl.offline_dataset_utils import load_metadata_rows

    try:
        episode_rows, mistake_rows = load_metadata_rows(dataset.root)
    except FileNotFoundError:
        return {}

    size = len(dataset)
    mistakes: set[int] = set()
    for row in mistake_rows:
        if row["mistake"]:
            mistakes.update(range(int(row["from_index"]), min(int(row["to_index"]), size)))

    lookup: dict[int, dict] = {}
    for row in episode_rows:
        for idx in range(int(row["from_index"]), min(int(row["to_index"]), size)):
            lookup[idx] = {"quality": int(row["quality"]), "mistake": idx in mistakes}
    return lookup


def get_frame_data(dataset, global_idx: int, chunk_size: int):
    """
    Pull a single frame + its GT action chunk from a LeRobot dataset.

    Returns:
        obs:         dict[str, Tensor] with batch dim 1, keys starting with "observation."
        gt_actions:  Tensor [chunk_size, action_dim] — raw (unnormalised), pad with last action
        state:       Tensor [state_dim] or None — current joint state if available
        gt_subtask:  str — subtask description from metadata ("" if absent)
        task_str:    str — high-level task
        episode_idx: int
        frame_idx:   int
    """
    frame = dataset[global_idx]
    episode_idx = frame["episode_index"].item()
    frame_idx = frame["frame_index"].item()
    task_str = frame.get("task", "")

    gt_actions = []
    for offset in range(chunk_size):
        candidate_idx = global_idx + offset
        if candidate_idx >= len(dataset):
            break
        f_item = dataset.hf_dataset[candidate_idx]
        if f_item["episode_index"].item() != episode_idx:
            break
        is_pad = f_item.get("action_is_pad", False)
        if isinstance(is_pad, torch.Tensor):
            is_pad = is_pad.item()
        if is_pad:
            break
        gt_actions.append(f_item["action"].detach().clone())

    if not gt_actions:
        gt_actions = [torch.zeros_like(frame["action"])]
    while len(gt_actions) < chunk_size:
        gt_actions.append(gt_actions[-1].clone())
    gt_actions = torch.stack(gt_actions[:chunk_size])

    obs = {
        k: v.unsqueeze(0)
        for k, v in frame.items()
        if k.startswith("observation.") and isinstance(v, torch.Tensor)
    }

    state = None
    if "observation.state" in frame:
        state = frame["observation.state"].float()

    subtask_idx = get_subtask_idx(dataset, global_idx)
    gt_subtask = get_subtask_str(dataset, subtask_idx)

    return obs, gt_actions, state, gt_subtask, task_str, episode_idx, frame_idx


def get_action_chunk_lowdim(dataset, global_idx: int, chunk_size: int):
    """GT action chunk + state for one frame, without decoding video.

    `get_frame_data` opens `dataset[global_idx]`, which decodes every camera for a
    frame whose images are then discarded — ten thousand frames' worth when the
    action probe fits its reference manifold. Actions, state and episode columns all
    live in the parquet, so a reference pass never needs the video decoder.

    Chunks stop at the episode boundary and repeat the last action to length, matching
    `get_frame_data`.

    Returns ``(gt_actions [chunk_size, action_dim] raw, state [state_dim] or None,
    episode_idx, frame_idx)``.
    """
    row = dataset.hf_dataset[global_idx]
    episode_idx = int(row["episode_index"].item())
    frame_idx = int(row["frame_index"].item())

    actions = []
    for offset in range(chunk_size):
        candidate_idx = global_idx + offset
        if candidate_idx >= len(dataset):
            break
        item = dataset.hf_dataset[candidate_idx]
        if int(item["episode_index"].item()) != episode_idx:
            break
        actions.append(item["action"].detach().clone().float())

    while len(actions) < chunk_size:
        actions.append(actions[-1].clone())
    gt_actions = torch.stack(actions[:chunk_size])

    state = row["observation.state"].float() if "observation.state" in row else None
    return gt_actions, state, episode_idx, frame_idx


def probe_frame_inputs(
    dataset,
    cfg,
    global_idx: int,
    chunk_size: int,
    *,
    with_depth: bool = True,
    with_history: bool = True,
    metadata: dict | None = DEPLOYMENT_METADATA,
) -> dict:
    """One frame in the deployment regime: observation + wrist depth + short-term
    history + the subtask and metadata clauses the action prompt carries at rollout.

    This is the prompt every probe should measure against. `get_frame_data` alone
    yields a prompt with no subtask, no metadata and no history — a regime that
    covers ~1.35% of training samples.

    ``global_idx`` must sit on the image/depth stride grid whenever ``with_depth``
    is set; `sample_episodes_evenly` and the action-inspector sampler snap for you.

    Returns a dict with ``obs``, ``gt_actions``, ``state``, ``subtask``, ``task``,
    ``metadata``, ``episode_idx``, ``frame_idx``, ``global_idx``.
    """
    from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png

    obs, gt_actions, state, gt_subtask, task_str, episode_idx, frame_idx = get_frame_data(
        dataset, global_idx, chunk_size
    )

    pointmap_cfg = getattr(cfg.policy, "pointmap_config", None)
    if with_depth and pointmap_cfg is not None:
        depth = load_depth_png(dataset.root, f"{pointmap_cfg.depth_key}.depth", episode_idx, frame_idx)
        obs[f"observation.depth.{pointmap_cfg.depth_key}"] = torch.from_numpy(
            depth.astype(np.float32)
        ).reshape(1, 1, *depth.shape)

    memory_cfg = getattr(cfg.policy, "memory", None)
    if with_history and memory_cfg is not None and memory_cfg.history_keys and memory_cfg.history_num_samples > 0:
        keys = [str(k) for k in memory_cfg.history_keys]
        if not (with_depth and pointmap_cfg is not None):
            keys = [k for k in keys if not k.startswith("depth.")]
        obs.update(assemble_frame_history(dataset, global_idx, memory_cfg, cfg.env.fps, keys))

    return {
        "obs": obs,
        "gt_actions": gt_actions,
        "state": state,
        "subtask": gt_subtask,
        "task": task_str,
        "metadata": None if metadata is None else dict(metadata),
        "episode_idx": episode_idx,
        "frame_idx": frame_idx,
        "global_idx": global_idx,
    }


def history_offsets(memory_cfg, fps: float) -> list[int]:
    """Lookback distances in frames, oldest → newest (buffer.py
    `_normalize_history_offsets`: sorted descending, deduplicated)."""
    n = memory_cfg.history_num_samples
    stride = memory_cfg.history_window_seconds * fps / n
    return sorted({round(stride * i) for i in range(1, n + 1)}, reverse=True)


def assemble_frame_history(
    dataset, global_idx: int, memory_cfg, fps: float, keys: list[str], *, depth_root=None
) -> dict:
    """Gather the short-term lookback window for one dataset frame, matching the
    ReplayBuffer's oldest→newest slots (buffer.py `_gather_history` /
    `_normalize_history_offsets`): offsets sorted descending, and invalid slots that
    reach before the episode start repeat the episode's first frame (the π0.7
    repeat-pad rule).

    Depth keys (``depth.<cam>.depth``) live in the PNG16 sidecar rather than in the
    dataset columns, so they are read with ``load_depth_png`` against ``depth_root``
    (defaults to ``dataset.root``). Their slots must land on the sidecar's stride
    grid; callers sample stride-snapped anchors and the offsets are multiples of the
    stride, so only the episode-start clamp can move a slot, and frame 0 is on-grid.

    Returns ``{f"history.{key}": tensor (1, T_h, ...)}`` for each requested key,
    ready to drop into the obs dict — ``batch_to_transition`` routes ``history.*``
    to COMPLEMENTARY_DATA, where the MolmoAct2 pack step and the point-map encoder
    read it. Requesting a key that resolves to nothing is a warning, not a silent
    drop: that is how depth history went missing from every probe.
    """
    from lerobot.scripts.lerobot_memmap_buffer_cache import load_depth_png

    offsets = history_offsets(memory_cfg, fps)
    frame = dataset[global_idx]
    frame_idx = int(frame["frame_index"].item())
    episode_idx = int(frame["episode_index"].item())
    ep_start = global_idx - frame_idx  # first global index of this episode

    depth_keys = [key for key in keys if str(key).startswith("depth.")]
    frame_keys = [key for key in keys if not str(key).startswith("depth.")]

    out: dict[str, torch.Tensor] = {}

    if frame_keys:
        slot_frames = [dataset[max(global_idx - o, ep_start)] for o in offsets]
        for key in frame_keys:
            if key not in slot_frames[0]:
                logging.warning(f"[probe] history key {key!r} is not a dataset column — window omitted.")
                continue
            out[f"history.{key}"] = torch.stack([f[key] for f in slot_frames]).unsqueeze(0)

    for key in depth_keys:
        sidecar = str(key)[len("depth."):]  # "depth.wrist.depth" -> "wrist.depth"
        slots = [
            torch.from_numpy(
                load_depth_png(
                    depth_root if depth_root is not None else dataset.root,
                    sidecar,
                    episode_idx,
                    max(frame_idx - o, 0),
                ).astype(np.float32)
            )
            for o in offsets
        ]
        out[f"history.{key}"] = torch.stack(slots).unsqueeze(0)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Episode / frame sampling
# ──────────────────────────────────────────────────────────────────────────────

def _snap_position(dataset, indices: list[int], pos: int, stride: int) -> tuple[int, int, int]:
    """Move a within-episode position back onto the stride grid.

    Returns ``(pos, frame_idx, global_idx)``. Image and depth rows are only stored
    every ``stride``-th frame (buffer.py `_gather_history`, and the depth sidecar
    has no PNG in between), so an off-grid anchor evaluates a frame the model was
    never trained on and cannot be given depth at all.
    """
    global_idx = indices[pos]
    fr_idx = int(dataset.hf_dataset[global_idx]["frame_index"].item())
    back = fr_idx % stride
    if back:
        pos = max(pos - back, 0)
        global_idx = indices[pos]
        fr_idx = int(dataset.hf_dataset[global_idx]["frame_index"].item())
    return pos, fr_idx, global_idx


def sample_episodes_evenly(
    dataset,
    n_per_episode: int,
    max_episodes: Optional[int],
    seed: int,
    stride: int = 1,
) -> list[tuple[int, int, int]]:
    """
    Sample *n_per_episode* evenly-spaced frames from each episode.

    If *max_episodes* is set, draw a reproducible random subset of episodes.
    ``stride`` snaps every anchor onto the image/depth grid (`policy.image_stride`);
    frames that collide after snapping are dropped, so a request finer than the
    grid returns fewer samples rather than duplicates.

    Returns list of (episode_idx, frame_idx_in_episode, global_idx).
    """
    ep_to_indices = build_episode_index(dataset)
    episodes = sorted(ep_to_indices.keys())
    if max_episodes is not None:
        rng = np.random.RandomState(seed)
        episodes = sorted(
            rng.choice(episodes, size=min(max_episodes, len(episodes)), replace=False).tolist()
        )

    samples: list[tuple[int, int, int]] = []
    for ep_idx in episodes:
        indices = ep_to_indices[ep_idx]
        n = min(n_per_episode, len(indices))
        seen: set[int] = set()
        for pos in np.linspace(0, len(indices) - 1, n, dtype=int):
            _, fr_idx, global_idx = _snap_position(dataset, indices, int(pos), stride)
            if global_idx in seen:
                continue
            seen.add(global_idx)
            samples.append((ep_idx, fr_idx, global_idx))

    return samples


def action_inspector_sample_seed(base_seed: int, global_idx: int, sample_idx: int = 0) -> int:
    """Stable independent noise seed; sample 0 is the Action Inspector's scored draw."""
    return int(base_seed) + int(global_idx) + int(sample_idx) * 1_000_003


def sample_action_inspector_frames(dataset, probe_cfg, stride: int = 1) -> list[tuple[int, int, int]]:
    """Anchor frames for the Action Inspector.

    Fixed time stride, optional episode allow-list, per-episode cap, returned in the
    common ``(episode_idx, frame_idx, global_idx)`` shape the other probes use.

    Every frame is snapped to the stored image/depth grid. ``trace_anchor_stride_s``
    is converted through the dataset FPS rather than the environment FPS because
    these are dataset coordinates.
    """
    ep_to_indices = build_episode_index(dataset)
    wanted = {
        int(episode)
        for episode in (getattr(probe_cfg, "trace_episodes", None) or "").split(",")
        if episode.strip()
    }
    stride = max(int(stride), 1)
    frame_stride = max(
        int(round(float(probe_cfg.trace_anchor_stride_s) * float(dataset.fps))),
        1,
    )
    frame_stride -= frame_stride % stride
    frame_stride = max(frame_stride, stride)

    samples: list[tuple[int, int, int]] = []
    for episode_idx in sorted(ep_to_indices):
        if wanted and episode_idx not in wanted:
            continue
        indices = ep_to_indices[episode_idx]
        for episode_samples, position in enumerate(
            range(0, len(indices), frame_stride), start=1
        ):
            _, frame_idx, global_idx = _snap_position(dataset, indices, position, stride)
            samples.append((episode_idx, frame_idx, global_idx))
            if episode_samples >= int(probe_cfg.trace_max_anchors_per_episode):
                break
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessor introspection
# ──────────────────────────────────────────────────────────────────────────────

def find_normalizer_step(preprocessor):
    """
    Return the NormalizerProcessorStep from a preprocessor pipeline.

    Duck-typed (``norm_map`` + ``_tensor_stats``) so the same lookup works for
    pi05's ``NormalizerProcessorStep`` and molmoact2's
    ``MolmoAct2MaskedNormalizerProcessorStep`` (both subclass the base
    NormalizerProcessorStep).
    """
    for step in preprocessor.steps:
        if hasattr(step, "norm_map") and hasattr(step, "_tensor_stats"):
            return step
    raise RuntimeError(
        f"No normalizer step found in preprocessor pipeline. "
        f"Steps: {[type(s).__name__ for s in preprocessor.steps]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction
# ──────────────────────────────────────────────────────────────────────────────

def run_pca(X: torch.Tensor, n_components: int, label: str, pca_dir: str):
    """
    Fit PCA on *X* (N, D). Saves a two-panel scree plot.

    Returns (X_pca tensor float32, fitted sklearn PCA).
    """
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    with threadpool_limits(limits=1, user_api="blas"):
        X_pca = pca.fit_transform(X.numpy())

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    comp90 = int(np.searchsorted(cumvar, 0.90)) + 1
    comp95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_,
                color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"{label} — per-component variance")

    axes[1].plot(range(1, n_components + 1), cumvar, color="steelblue", linewidth=1.5)
    axes[1].axhline(0.90, color="#888", linestyle="--", linewidth=0.9, label=f"90% @ {comp90}")
    axes[1].axhline(0.95, color="orange", linestyle="--", linewidth=0.9, label=f"95% @ {comp95}")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title(f"{label} — cumulative variance")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.02)

    fig.suptitle(
        f"PCA scree — {label}  ({X.shape[0]} samples, {X.shape[1]} dims → top {n_components})",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = os.path.join(pca_dir, f"{label}_pca_scree.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return torch.from_numpy(X_pca.astype(np.float32)), pca


def run_umap(X_pca: torch.Tensor, n_components: int, n_neighbors: int,
             min_dist: float, seed: int) -> np.ndarray:
    """
    In-sample UMAP fit_transform. Returns (N, n_components) array.

    For out-of-sample projection (action probe), use umap.UMAP().fit() then
    reducer.transform() directly.
    """
    import umap as umap_lib

    reducer = umap_lib.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden", category=UserWarning)
        return reducer.fit_transform(X_pca.numpy())


# ──────────────────────────────────────────────────────────────────────────────
# 2D matplotlib helpers
# ──────────────────────────────────────────────────────────────────────────────

def panel_caption(ax, lines: list[str], y: float = -0.185) -> None:
    """The computation behind the panel, printed under the panel.

    Pre-wrapped rather than filled, because the lines carry mathtext spans that a
    wrapper would break mid-formula.
    """
    ax.text(0.0, y, "\n".join(lines), transform=ax.transAxes, fontsize=7.4,
            va="top", ha="left", color="#333333", linespacing=1.55)


def ax_style(ax, title: str, width: int = 60) -> None:
    """Consistent axis styling for 2D UMAP scatter / trajectory plots."""
    ax.set_title(textwrap.fill(title, width=width), fontsize=9)
    ax.set_xlabel("UMAP-1", fontsize=8)
    ax.set_ylabel("UMAP-2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")


# ──────────────────────────────────────────────────────────────────────────────
# Plotly 3D helpers
# ──────────────────────────────────────────────────────────────────────────────

def plotly_3d_layout(title: str) -> dict:
    """Layout dict for a clean Plotly 3D scatter."""
    return dict(
        title=dict(text=title, font=dict(size=13)),
        scene=dict(
            xaxis=dict(title="UMAP-1", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            yaxis=dict(title="UMAP-2", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            zaxis=dict(title="UMAP-3", showgrid=True, gridcolor="#e0e0e0", gridwidth=1),
            bgcolor="white",
            aspectmode="auto",
        ),
        paper_bgcolor="white",
        legend=dict(font=dict(size=12), itemsizing="constant",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc", borderwidth=1),
        margin=dict(l=0, r=0, b=0, t=55),
        height=720,
    )


def frame_colors_rgba(frames, cmap_name: str, alpha: float = 0.85) -> list[str]:
    """
    Map frame indices to rgba color strings via a sequential colormap.
    Convention: dark = early frame, pale = late frame.
    """
    frames = np.asarray(frames, dtype=float)
    fmin, fmax = frames.min(), frames.max()
    norm = 0.9 - (frames - fmin) / max(fmax - fmin, 1.0) * 0.6
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(norm)
    return [f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})" for r, g, b, _ in rgba]
