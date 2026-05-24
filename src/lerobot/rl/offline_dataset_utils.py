from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F_vision

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.utils.constants import ACTION


def _get_additional_dataset_paths(cfg) -> list[str]:
    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is None:
        return []
    paths = getattr(dataset_cfg, "additional_offline_dataset_paths", None)
    if not paths:
        return []
    return [str(p) for p in paths]


def _expected_image_hw(cfg) -> tuple[int, int] | None:
    if hasattr(cfg.policy, "image_storage_size"):
        size = getattr(cfg.policy, "image_storage_size")
        if size is None:
            return None
        return int(size[0]), int(size[1])
    for key, feature in getattr(cfg.policy, "input_features", {}).items():
        if "image" not in key:
            continue
        shape = getattr(feature, "shape", None)
        if shape is None and isinstance(feature, dict):
            shape = feature.get("shape")
        if shape is not None and len(shape) >= 2:
            return int(shape[-2]), int(shape[-1])
    return None


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


def _move_transition_data(data: dict[str, Any], storage_device, image_hw: tuple[int, int] | None) -> None:
    for key, value in list(data.items()):
        if isinstance(value, dict):
            for subkey, tensor in list(value.items()):
                if not isinstance(tensor, torch.Tensor):
                    continue
                if "images" in subkey and image_hw is not None and tensor.shape[-2:] != image_hw:
                    tensor = F_vision.resize(tensor, image_hw).clamp(0.0, 1.0)
                value[subkey] = tensor.to(device=storage_device)
        elif isinstance(value, torch.Tensor):
            data[key] = value.to(device=storage_device)


def _remap_complementary_info(comp_info: dict, remap_table: dict[int, int]) -> dict:
    if "subtask_index" in comp_info:
        old_idx = comp_info["subtask_index"]
        scalar_old_idx = int(old_idx.item()) if isinstance(old_idx, torch.Tensor) else int(old_idx)
        if scalar_old_idx in remap_table:
            new_idx = remap_table[scalar_old_idx]
            if isinstance(old_idx, torch.Tensor):
                comp_info["subtask_index"] = torch.tensor(new_idx, dtype=old_idx.dtype, device=old_idx.device)
            else:
                comp_info["subtask_index"] = new_idx
    return comp_info


def load_additional_offline_datasets(
    cfg,
    offline_dataset,
    offline_replay_buffer: ReplayBuffer,
    storage_device,
    is_main_process: bool = True,
) -> None:
    """Load configured extra LeRobot datasets into an existing offline ReplayBuffer.

    Config convention, inherited from the PI05 path:
        cfg.dataset.additional_offline_dataset_paths: list[str]

    Each additional dataset uses cfg.dataset.repo_id with a different local root.
    Subtask indices are remapped into the main dataset metadata when both datasets
    expose `meta.subtasks`. Additional datasets are marked `is_golden=False` unless
    they already provide that complementary field.
    """
    paths = _get_additional_dataset_paths(cfg)
    if not paths:
        return

    image_hw = _expected_image_hw(cfg)
    repo_id = getattr(cfg.dataset, "repo_id", None)
    state_keys = cfg.policy.input_features.keys()

    for dataset_path in paths:
        if is_main_process:
            logging.info("[AdditionalOffline] Loading dataset from %s", dataset_path)

        additional_dataset = LeRobotDataset(repo_id=repo_id, root=Path(dataset_path))
        additional_dataset.delta_timestamps = None
        additional_dataset.delta_indices = None

        generator = ReplayBuffer._lerobotdataset_to_transitions_generator(
            additional_dataset,
            state_keys=state_keys,
        )
        remap_table = remap_subtasks_for_dataset(offline_dataset, additional_dataset, is_main_process)

        if is_main_process:
            logging.info("[AdditionalOffline] Adding transitions from %s", dataset_path)

        for data in generator:
            _move_transition_data(data, storage_device, image_hw)
            comp_info = data.get("complementary_info") or {}
            comp_info = _remap_complementary_info(comp_info, remap_table)
            comp_info.setdefault("is_golden", False)

            offline_replay_buffer.add(
                state=data["state"],
                action=data[ACTION],
                reward=data["reward"],
                next_state=data["next_state"],
                done=data["done"],
                truncated=data.get("truncated", False),
                complementary_info=comp_info,
            )

        if is_main_process:
            logging.info(
                "[AdditionalOffline] Finished %s. Buffer size: %s",
                dataset_path,
                len(offline_replay_buffer),
            )
