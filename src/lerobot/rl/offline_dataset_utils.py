from __future__ import annotations

import logging
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer, concatenate_batch_transitions
from lerobot.utils.constants import ACTION


def _get_additional_dataset_paths(cfg) -> list[str]:
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return []
    paths = getattr(dataset_cfg, "additional_offline_dataset_paths", None)
    if not paths:
        return []
    return [str(p) for p in paths]


def _idx_to_subtask_name(dataset) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not (hasattr(dataset, "meta") and hasattr(dataset.meta, "subtasks")):
        return mapping
    df = dataset.meta.subtasks
    if not hasattr(df, "columns") or "subtask_index" not in df.columns:
        return mapping
    for idx, row in df.iterrows():
        name = idx if isinstance(idx, str) else row.get("subtask", str(idx))
        mapping[int(row["subtask_index"])] = name
    return mapping


def remap_subtasks_for_dataset(target_dataset, source_dataset, is_main_process: bool = False) -> dict[int, int]:
    """Map source subtask indices into the target dataset subtask index space."""
    remap_table: dict[int, int] = {}
    target_idx_to_name = _idx_to_subtask_name(target_dataset)
    source_idx_to_name = _idx_to_subtask_name(source_dataset)
    if not source_idx_to_name:
        return remap_table

    target_name_to_idx = {name: idx for idx, name in target_idx_to_name.items()}
    next_new_index = max(target_idx_to_name.keys()) + 1 if target_idx_to_name else 0

    for old_idx, name in source_idx_to_name.items():
        if name in target_name_to_idx:
            remap_table[old_idx] = target_name_to_idx[name]
            continue

        new_idx = next_new_index
        remap_table[old_idx] = new_idx
        target_name_to_idx[name] = new_idx
        target_idx_to_name[new_idx] = name
        next_new_index += 1

        if hasattr(target_dataset, "meta") and hasattr(target_dataset.meta, "subtasks"):
            import pandas as pd

            new_row = pd.DataFrame([{"subtask_index": new_idx}], index=[name])
            target_dataset.meta.subtasks = pd.concat([target_dataset.meta.subtasks, new_row])

    if is_main_process and remap_table:
        logging.info("[AdditionalOffline] Remapping subtasks: %s", remap_table)
    return remap_table


def load_additional_offline_buffers(
    cfg,
    main_dataset,
    device,
    storage_device,
    is_main_process: bool = True,
    history_offsets: dict[str, list[int]] | None = None,
) -> list[ReplayBuffer]:
    """Load extra offline datasets as independent ReplayBuffers.

    Each returned buffer is sized to its own dataset (memmap-backed when a
    matching cache exists under cfg.buffer_cache_dir), so loading does not
    grow the main offline buffer or touch its memmap. Sampling across the
    main + additional buffers is the caller's job — see
    `make_combined_offline_iterator`.

    Subtask indices are remapped into the main dataset's subtask vocabulary
    in-place on each additional buffer's RAM-resident `complementary_info`.
    `is_golden` is forced to False on every additional buffer (the cache may
    have it baked in as True from --inject-golden — we overwrite).
    """
    paths = _get_additional_dataset_paths(cfg)
    if not paths:
        return []

    repo_id = getattr(cfg.dataset, "repo_id", None)
    state_keys = list(cfg.policy.input_features.keys())
    cache_dir = getattr(cfg, "buffer_cache_dir", None)
    image_storage_dtype = getattr(cfg.policy, "image_storage_dtype", "bfloat16")
    image_storage_size = getattr(cfg.policy, "image_storage_size", None)

    buffers: list[ReplayBuffer] = []
    for dataset_path in paths:
        if is_main_process:
            logging.info("[AdditionalOffline] Loading dataset from %s", dataset_path)

        additional_dataset = LeRobotDataset(repo_id=repo_id, root=Path(dataset_path))
        additional_dataset.delta_timestamps = None
        additional_dataset.delta_indices = None

        remap_table = remap_subtasks_for_dataset(main_dataset, additional_dataset, is_main_process)

        cached = None
        if cache_dir is not None:
            cached = ReplayBuffer.find_cache(
                additional_dataset,
                cache_dir,
                state_keys=state_keys,
                image_storage_dtype=image_storage_dtype,
                image_storage_size=image_storage_size,
            )

        if cached is not None:
            if is_main_process:
                logging.info("[AdditionalOffline] Found memmap cache at %s; loading from disk", cached)
            buf = ReplayBuffer.from_cache(
                cache_dir=cached,
                device=device,
                use_drq=False,
                history_offsets=history_offsets,
            )
        else:
            if is_main_process:
                logging.info(
                    "[AdditionalOffline] No memmap cache for %s under %s; falling back to video decode. "
                    "Run lerobot_memmap_buffer_cache.py to build the cache for fast loads.",
                    dataset_path,
                    cache_dir,
                )
            buf = ReplayBuffer.from_lerobot_dataset(
                additional_dataset,
                device=device,
                state_keys=state_keys,
                storage_device=storage_device,
                optimize_memory=True,
                inject_complementary_info={"is_golden": False},
                image_storage_dtype=image_storage_dtype,
                image_storage_size=image_storage_size,
                history_offsets=history_offsets,
            )

        # Remap subtask indices on the (RAM-resident) complementary_info tensor.
        if (
            remap_table
            and buf.has_complementary_info
            and "subtask_index" in buf.complementary_info
        ):
            ci = buf.complementary_info["subtask_index"]
            for old_idx, new_idx in remap_table.items():
                ci[ci == old_idx] = new_idx

        # Force is_golden=False (cache may carry True from --inject-golden).
        ci_dict = buf.complementary_info
        if "is_golden" in ci_dict:
            ci_dict["is_golden"].fill_(0)
        else:
            existing = next(iter(ci_dict.values()), None)
            ig_device = existing.device if existing is not None else storage_device
            ci_dict["is_golden"] = torch.zeros(buf.size, dtype=torch.bfloat16, device=ig_device)
            if "is_golden" not in buf.complementary_info_keys:
                buf.complementary_info_keys.append("is_golden")
            buf.has_complementary_info = True

        if is_main_process:
            logging.info(
                "[AdditionalOffline] Loaded %s — %d transitions",
                dataset_path,
                buf.size,
            )
        buffers.append(buf)

    return buffers


def make_combined_offline_iterator(
    buffers: list[ReplayBuffer],
    batch_size: int,
    async_prefetch: bool = True,
    queue_size: int = 2,
    action_chunk_size: int = 50,
):
    """Yield batches drawn equally from every buffer (main + additionals).

    Each call to `next(...)` produces a batch of `batch_size` transitions split
    evenly: every buffer contributes `batch_size // N`, and the first
    `batch_size % N` get one extra. Per-buffer iterators preserve their own
    async prefetch threads, so this composes with `get_iterator`'s prefetch.
    """
    if not buffers:
        raise ValueError("make_combined_offline_iterator needs at least one buffer.")

    n = len(buffers)
    per_buf = [batch_size // n] * n
    for i in range(batch_size % n):
        per_buf[i] += 1

    iters = [
        b.get_iterator(
            batch_size=per_buf[i],
            async_prefetch=async_prefetch,
            queue_size=queue_size,
            action_chunk_size=action_chunk_size,
        )
        for i, b in enumerate(buffers)
    ]

    while True:
        batch = next(iters[0])
        for it in iters[1:]:
            action_dim = batch[ACTION].shape[-1]
            batch = concatenate_batch_transitions(batch, next(it), action_dim=action_dim)
        yield batch
