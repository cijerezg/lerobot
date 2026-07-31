from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass(frozen=True)
class OfflineDatasetSource:
    """Resolved runtime description of one physical offline dataset."""

    name: str
    root: str | None
    repo_id: str
    weight: float
    normalization_source: bool = False
    episodes: list[int] | None = None


def _source_field(source, name: str, default=None):
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def get_offline_dataset_sources(cfg) -> list[OfflineDatasetSource]:
    """Resolve explicit collection sources or the legacy primary + additional paths."""
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return []
    configured = list(getattr(dataset_cfg, "sources", None) or [])
    legacy_paths = list(getattr(dataset_cfg, "additional_offline_dataset_paths", None) or [])
    if configured and legacy_paths:
        raise ValueError("Use dataset.sources or additional_offline_dataset_paths, not both.")
    fallback_repo_id = str(getattr(dataset_cfg, "repo_id", None) or "local/dataset")
    if configured:
        marked = [
            i for i, source in enumerate(configured) if _source_field(source, "normalization_source", False)
        ]
        if len(marked) > 1:
            raise ValueError("Exactly one dataset source may set normalization_source=true.")
        primary_index = marked[0] if marked else 0
        ordered = [configured[primary_index], *(s for i, s in enumerate(configured) if i != primary_index)]
        resolved = []
        for i, source in enumerate(ordered):
            root = str(_source_field(source, "root"))
            repo_id = str(_source_field(source, "repo_id", None) or fallback_repo_id)
            name = _source_field(source, "name", None) or Path(root).name
            weight = float(_source_field(source, "weight", 1.0))
            if weight <= 0:
                raise ValueError(f"Dataset source {name!r} has non-positive weight {weight}.")
            resolved.append(
                OfflineDatasetSource(
                    name=str(name),
                    root=root,
                    repo_id=repo_id,
                    weight=weight,
                    normalization_source=i == 0,
                    episodes=_source_field(source, "episodes", None),
                )
            )
        return resolved
    root = getattr(dataset_cfg, "root", None)
    primary_name = Path(root).name if root else fallback_repo_id.split("/")[-1]
    sources = [
        OfflineDatasetSource(
            name=primary_name,
            root=str(root) if root is not None else None,
            repo_id=fallback_repo_id,
            weight=1.0,
            normalization_source=True,
            episodes=getattr(dataset_cfg, "episodes", None),
        )
    ]
    sources.extend(
        OfflineDatasetSource(
            name=Path(path).name,
            root=str(path),
            repo_id=fallback_repo_id,
            weight=1.0,
        )
        for path in legacy_paths
    )
    return sources


def load_offline_dataset(cfg, source: OfflineDatasetSource) -> LeRobotDataset:
    """Load one collection source without concatenating it with its peers."""
    from lerobot.transforms import ImageTransforms

    dataset_cfg = cfg.dataset
    image_transforms = (
        ImageTransforms(dataset_cfg.image_transforms) if dataset_cfg.image_transforms.enable else None
    )
    episodes = source.episodes
    if episodes is None and source.normalization_source and dataset_cfg.max_episodes is not None:
        episodes = list(range(dataset_cfg.max_episodes))
    dataset = LeRobotDataset(
        source.repo_id,
        root=source.root,
        episodes=episodes,
        image_transforms=image_transforms,
        revision=dataset_cfg.revision,
        video_backend=dataset_cfg.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    return dataset


def _get_additional_dataset_paths(cfg) -> list[str]:
    return [source.root for source in get_offline_dataset_sources(cfg)[1:] if source.root is not None]


def get_offline_dataset_weights(cfg) -> list[float]:
    return [source.weight for source in get_offline_dataset_sources(cfg)]


def pool_lowdim_stats(cfg, dataset, is_main_process: bool = False) -> None:
    """Replace the normalization source's action/state stats with pooled ones.

    Only ``sources[0]`` supplies normalization stats, but normalized state and
    action are clamped to [-1, 1] downstream: quantiles from one dataset collapse
    whatever the other sources reach beyond them. Quantiles do not aggregate from
    per-dataset summaries, so recompute them exactly over the pooled low-dim
    columns — a few hundred thousand rows of two 7-vectors, cheap to read.
    Visual stats are untouched (VISUAL is IDENTITY-normalized).
    """
    import numpy as np
    import pyarrow.parquet as pq

    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES

    roots = [source.root for source in get_offline_dataset_sources(cfg) if source.root is not None]
    stats = getattr(getattr(dataset, "meta", None), "stats", None)
    if len(roots) < 2 or not stats:
        return

    for key in (ACTION, OBS_STATE):
        if key not in stats:
            continue
        columns = [
            pq.read_table(f, columns=[key]).to_pandas()[key]
            for root in roots
            for f in sorted(Path(root).rglob("data/**/*.parquet"))
        ]
        values = np.concatenate([np.stack(c.values) for c in columns]).astype(np.float32)
        pooled = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
            "count": np.array([len(values)]),
        }
        for q in DEFAULT_QUANTILES:
            pooled[f"q{int(q * 100):02d}"] = np.quantile(values, q, axis=0)
        if is_main_process:
            logging.info(
                "[OfflineCollection] Pooled %s stats over %d roots / %d frames: "
                "q01 %s -> %s",
                key,
                len(roots),
                len(values),
                np.round(stats[key]["q01"], 1),
                np.round(pooled["q01"], 1),
            )
        stats[key] = pooled


def load_summary_segments(root) -> tuple[list[dict], list[str]]:
    """Read meta/summaries.parquet (written by summary_annotate.py) into the inputs
    ReplayBuffer.materialize_summaries expects: segment rows sorted by
    (episode_index, segment_index) and the summary texts in the same order.
    Returns ([], []) when the dataset has no summaries."""
    path = Path(root) / "meta" / "summaries.parquet"
    if not path.exists():
        return [], []
    import pandas as pd

    df = pd.read_parquet(path).sort_values(["episode_index", "segment_index"])
    segments = df[["episode_index", "segment_index", "from_index", "to_index"]].to_dict("records")
    return segments, [str(s) for s in df["summary"]]


def load_metadata_rows(root) -> tuple[list[dict], list[dict]]:
    """Read meta/episode_metadata.parquet + meta/mistakes.parquet (written by
    metadata_annotate.py) into the inputs ReplayBuffer.materialize_metadata
    expects. Raises when missing: metadata_enabled requires annotated datasets."""
    meta = Path(root) / "meta"
    for name in ("episode_metadata.parquet", "mistakes.parquet"):
        if not (meta / name).exists():
            raise FileNotFoundError(
                f"metadata_enabled but {meta / name} is missing — run metadata_annotate.py on this dataset."
            )
    import pandas as pd

    episode_rows = pd.read_parquet(meta / "episode_metadata.parquet").to_dict("records")
    mistake_rows = pd.read_parquet(meta / "mistakes.parquet").to_dict("records")
    return episode_rows, mistake_rows


def _idx_to_name(dataset, table_name: str, index_column: str, text_column: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    meta = getattr(dataset, "meta", None)
    table = getattr(meta, table_name, None) if meta is not None else None
    if not hasattr(table, "columns") or index_column not in table.columns:
        return mapping
    for idx, row in table.iterrows():
        name = idx if isinstance(idx, str) else row.get(text_column, str(idx))
        mapping[int(row[index_column])] = str(name)
    return mapping


def _idx_to_task_name(dataset) -> dict[int, str]:
    return _idx_to_name(dataset, "tasks", "task_index", "task")


def _idx_to_subtask_name(dataset) -> dict[int, str]:
    return _idx_to_name(dataset, "subtasks", "subtask_index", "subtask")


def _label_key(name: str) -> str:
    """Conservative matching only; raw LLM wording remains the stored vocabulary text."""
    return " ".join(name.strip().split()).casefold().rstrip(".")


def _remap_vocabulary(
    target_dataset,
    source_dataset,
    *,
    table_name: str,
    index_column: str,
    text_column: str,
    is_main_process: bool = False,
) -> dict[int, int]:
    target = _idx_to_name(target_dataset, table_name, index_column, text_column)
    source = _idx_to_name(source_dataset, table_name, index_column, text_column)
    if not source:
        return {}
    key_to_index = {_label_key(name): idx for idx, name in target.items()}
    next_index = max(target, default=-1) + 1
    remap: dict[int, int] = {}
    new_rows: list[tuple[str, int]] = []
    for old_index, name in source.items():
        key = _label_key(name)
        if key in key_to_index:
            remap[old_index] = key_to_index[key]
        else:
            remap[old_index] = next_index
            key_to_index[key] = next_index
            new_rows.append((name, next_index))
            next_index += 1
    if new_rows:
        import pandas as pd

        meta = target_dataset.meta
        table = getattr(meta, table_name, None)
        additions = pd.DataFrame(
            {index_column: [index for _, index in new_rows]},
            index=pd.Index([name for name, _ in new_rows], name=text_column),
        )
        setattr(meta, table_name, additions if table is None else pd.concat([table, additions]))
    if is_main_process:
        logging.info("[OfflineCollection] %s remap: %s", table_name, remap)
    return remap


def remap_tasks_for_dataset(target_dataset, source_dataset, is_main_process: bool = False) -> dict[int, int]:
    return _remap_vocabulary(
        target_dataset,
        source_dataset,
        table_name="tasks",
        index_column="task_index",
        text_column="task",
        is_main_process=is_main_process,
    )


def remap_subtasks_for_dataset(
    target_dataset, source_dataset, is_main_process: bool = False
) -> dict[int, int]:
    return _remap_vocabulary(
        target_dataset,
        source_dataset,
        table_name="subtasks",
        index_column="subtask_index",
        text_column="subtask",
        is_main_process=is_main_process,
    )


def _dataset_index_column(dataset, key: str) -> torch.Tensor | None:
    hf_dataset = dataset.hf_dataset
    if key not in hf_dataset.column_names:
        return None
    values = hf_dataset.data.column(key).to_numpy(zero_copy_only=False).copy()
    return torch.as_tensor(values, dtype=torch.long)


def _subtask_indices_from_windows(dataset, num_frames: int) -> torch.Tensor | None:
    """Materialize reviewed subtask windows when no frame-level column exists.

    ``meta/subtask_windows.json`` stores global ``[from_index, to_index)`` frame
    ranges with subtask text. The text is resolved through
    ``meta/subtasks.parquet`` so the result uses the dataset-local vocabulary;
    collection remapping happens later in :func:`materialize_dataset_labels`.
    """
    root = getattr(dataset, "root", None)
    if root is None:
        return None
    path = Path(root) / "meta" / "subtask_windows.json"
    if not path.exists():
        return None

    with path.open() as f:
        payload = json.load(f)
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, dict):
        raise ValueError(f"{path} must contain an 'episodes' object.")

    index_to_name = _idx_to_subtask_name(dataset)
    if not index_to_name:
        raise ValueError(f"{path} exists but {Path(root) / 'meta' / 'subtasks.parquet'} is missing or empty.")
    name_to_index: dict[str, int] = {}
    for index, name in index_to_name.items():
        key = _label_key(name)
        existing = name_to_index.get(key)
        if existing is not None and existing != index:
            raise ValueError(
                f"Duplicate normalized subtask label {name!r} in {Path(root) / 'meta' / 'subtasks.parquet'}."
            )
        name_to_index[key] = index

    result = torch.full((num_frames,), -1, dtype=torch.long)
    assigned = torch.zeros(num_frames, dtype=torch.bool)
    episode_indices = _dataset_index_column(dataset, "episode_index")
    for raw_episode_index, windows in episodes.items():
        try:
            episode_index = int(raw_episode_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} has a non-integer episode key {raw_episode_index!r}.") from exc
        if not isinstance(windows, list):
            raise ValueError(f"{path} episode {episode_index} must contain a list of windows.")
        for window_index, window in enumerate(windows):
            if not isinstance(window, dict):
                raise ValueError(f"{path} episode {episode_index} window {window_index} must be an object.")
            try:
                start = int(window["from_index"])
                stop = int(window["to_index"])
                subtask = str(window["subtask"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} episode {episode_index} window {window_index} must define "
                    "integer from_index/to_index and a subtask."
                ) from exc
            if not (0 <= start < stop <= num_frames):
                raise ValueError(
                    f"{path} episode {episode_index} window {window_index} has invalid "
                    f"frame range [{start}, {stop}) for a {num_frames}-frame dataset."
                )
            if assigned[start:stop].any():
                raise ValueError(
                    f"{path} episode {episode_index} window {window_index} overlaps an earlier "
                    f"window in frame range [{start}, {stop})."
                )
            if episode_indices is not None and not torch.all(episode_indices[start:stop] == episode_index):
                raise ValueError(
                    f"{path} episode {episode_index} window {window_index} range [{start}, {stop}) "
                    "contains frames from another episode."
                )
            subtask_index = name_to_index.get(_label_key(subtask))
            if subtask_index is None:
                raise ValueError(
                    f"{path} episode {episode_index} window {window_index} uses unknown "
                    f"subtask {subtask!r}; add it to meta/subtasks.parquet."
                )
            result[start:stop] = subtask_index
            assigned[start:stop] = True
    return result


def _remap_indices(values: torch.Tensor, remap: dict[int, int]) -> torch.Tensor:
    result = torch.full_like(values, -1)
    for old_index, new_index in remap.items():
        result[values == old_index] = new_index
    return result


def _install_buffer_column(
    buffer: ReplayBuffer, key: str, values: torch.Tensor, fill_value: int = -1
) -> None:
    """Install a dataset-order label tensor in the replay buffer's physical slot order."""
    target = torch.full(
        (buffer.capacity, *values.shape[1:]),
        fill_value,
        dtype=values.dtype,
        device=buffer.storage_device,
    )
    start = max(0, len(values) - buffer.capacity)
    source = values[start:].to(buffer.storage_device)
    slots = torch.arange(start, len(values), device=buffer.storage_device) % buffer.capacity
    target[slots] = source
    buffer.complementary_info[key] = target
    if key not in buffer.complementary_info_keys:
        buffer.complementary_info_keys.append(key)
    buffer.has_complementary_info = True


def materialize_dataset_labels(
    buffer: ReplayBuffer,
    dataset,
    vocabulary_dataset,
    source_index: int,
    is_main_process: bool = False,
) -> None:
    """Overlay current task/subtask labels on heavy cache data without decoding video."""
    task_indices = _dataset_index_column(dataset, "task_index")
    if task_indices is None:
        raise ValueError(f"Offline dataset {dataset.root} has no task_index column.")
    task_remap = remap_tasks_for_dataset(vocabulary_dataset, dataset, is_main_process)
    _install_buffer_column(buffer, "task_index", _remap_indices(task_indices, task_remap))
    subtask_indices = _dataset_index_column(dataset, "subtask_index")
    if subtask_indices is None:
        subtask_indices = _subtask_indices_from_windows(dataset, len(task_indices))
        if subtask_indices is not None and is_main_process:
            logging.info(
                "[OfflineCollection] Materialized subtask_index from %s",
                Path(dataset.root) / "meta" / "subtask_windows.json",
            )
    if subtask_indices is not None:
        subtask_remap = remap_subtasks_for_dataset(vocabulary_dataset, dataset, is_main_process)
        _install_buffer_column(buffer, "subtask_index", _remap_indices(subtask_indices, subtask_remap))
    _install_buffer_column(
        buffer,
        "source_index",
        torch.full_like(task_indices, int(source_index)),
        fill_value=-1,
    )


def resolve_task_strings(raw_batch: dict, dataset, fallback_task: str, batch_size: int) -> list[str]:
    """Hydrate collection-global task indices into per-sample prompt strings."""
    complementary = raw_batch.get("complementary_info") or {}
    indices = complementary.get("task_index")
    mapping = _idx_to_task_name(dataset)
    if indices is None or not mapping:
        return [fallback_task] * batch_size
    flat = torch.as_tensor(indices).detach().cpu().reshape(-1).long().tolist()
    if len(flat) != batch_size:
        raise ValueError(f"Expected {batch_size} task indices, got {len(flat)}.")
    return [mapping.get(index, fallback_task) if index >= 0 else fallback_task for index in flat]


def load_additional_offline_buffers(
    cfg,
    main_dataset,
    device,
    storage_device,
    is_main_process: bool = True,
    history_offsets: dict[str, list[int]] | None = None,
    memory_cfg=None,
) -> list[ReplayBuffer]:
    """Load collection peers as independent cached buffers with fresh label overlays."""
    sources = get_offline_dataset_sources(cfg)[1:]
    if not sources:
        return []
    state_keys = list(cfg.policy.input_features.keys())
    cache_dir = getattr(cfg, "buffer_cache_dir", None)
    cache_policy = getattr(cfg, "cache_policy", "fallback")
    image_storage_dtype = getattr(cfg.policy, "image_storage_dtype", "bfloat16")
    image_storage_size = getattr(cfg.policy, "image_storage_size", None)
    image_stride = getattr(cfg.policy, "image_stride", 1)
    buffers: list[ReplayBuffer] = []
    for source_index, source in enumerate(sources, start=1):
        if is_main_process:
            logging.info(
                "[OfflineCollection] Loading %s from %s (weight=%s)",
                source.name,
                source.root,
                source.weight,
            )
        dataset = load_offline_dataset(cfg, source)
        cached = None
        if cache_dir is not None:
            cached = ReplayBuffer.find_cache(
                dataset,
                cache_dir,
                state_keys=state_keys,
                image_storage_dtype=image_storage_dtype,
                image_storage_size=image_storage_size,
                image_stride=image_stride,
            )
        if cached is None and cache_policy == "require":
            raise FileNotFoundError(
                f"No matching replay cache for source {source.name!r} ({source.root}) under "
                f"{cache_dir!r}. Build it with lerobot_memmap_buffer_cache.py using the "
                "same image storage dtype, size, and stride."
            )
        if cached is not None:
            if is_main_process:
                logging.info("[OfflineCollection] Found memmap cache at %s", cached)
            buffer = ReplayBuffer.from_cache(
                cache_dir=cached,
                device=device,
                use_drq=False,
                history_offsets=history_offsets,
            )
        else:
            if is_main_process:
                logging.info(
                    "[OfflineCollection] No cache for %s; falling back to video decode",
                    source.name,
                )
            buffer = ReplayBuffer.from_lerobot_dataset(
                dataset,
                device=device,
                state_keys=state_keys,
                storage_device=storage_device,
                optimize_memory=True,
                image_storage_dtype=image_storage_dtype,
                image_storage_size=image_storage_size,
                image_stride=image_stride,
                history_offsets=history_offsets,
            )
        materialize_dataset_labels(buffer, dataset, main_dataset, source_index, is_main_process)
        buffer.dataset = dataset
        buffer.offline_source = source
        if memory_cfg is not None and memory_cfg.metadata_enabled:
            buffer.materialize_metadata(*load_metadata_rows(dataset.root))
        buffers.append(buffer)
        if is_main_process:
            logging.info("[OfflineCollection] Loaded %s: %d transitions", source.name, buffer.size)
    return buffers


def _weighted_batch_sizes(batch_size: int, weights: list[float]) -> list[int]:
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError(f"Offline dataset weights must all be positive, got {weights}.")
    if batch_size < len(weights):
        raise ValueError(
            f"batch_size={batch_size} must be at least the number of dataset sources={len(weights)}."
        )
    remaining = batch_size - len(weights)
    total = sum(weights)
    quotas = [remaining * weight / total for weight in weights]
    extras = [int(quota) for quota in quotas]
    leftover = remaining - sum(extras)
    order = sorted(range(len(weights)), key=lambda i: quotas[i] - extras[i], reverse=True)
    for i in order[:leftover]:
        extras[i] += 1
    return [1 + extra for extra in extras]


def make_combined_offline_iterator(
    buffers: list[ReplayBuffer],
    batch_size: int,
    async_prefetch: bool = True,
    queue_size: int = 2,
    action_chunk_size: int = 50,
    weights: list[float] | None = None,
):
    """Yield fixed-size batches drawn from independent buffers by source weight."""
    if not buffers:
        raise ValueError("make_combined_offline_iterator needs at least one buffer.")
    if weights is None:
        weights = [1.0] * len(buffers)
    if len(weights) != len(buffers):
        raise ValueError(f"Expected {len(buffers)} weights, got {len(weights)}.")
    per_buffer = _weighted_batch_sizes(batch_size, weights)
    logging.info("[OfflineCollection] Batch allocation: %s (weights=%s)", per_buffer, weights)
    iterators = [
        buffer.get_iterator(
            batch_size=per_buffer[i],
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            action_chunk_size=action_chunk_size,
        )
        for i, buffer in enumerate(buffers)
    ]
    while True:
        batch = next(iterators[0])
        for iterator in iterators[1:]:
            action_dim = batch[ACTION].shape[-1]
            batch = concatenate_batch_transitions(batch, next(iterator), action_dim=action_dim)
        yield batch
