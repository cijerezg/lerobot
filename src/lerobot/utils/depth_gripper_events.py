"""Strict loading for materialized depth gripper-event supervision."""

from __future__ import annotations

import weakref
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

DEPTH_GRIPPER_EVENT_LABEL_FILENAME = "depth_gripper_event_labels.parquet"
DEPTH_GRIPPER_CLOSE_TARGET = "depth_gripper_close_target"
DEPTH_GRIPPER_OPEN_TARGET = "depth_gripper_open_target"
DEPTH_GRIPPER_EVENT_TARGET_KEYS = (
    DEPTH_GRIPPER_CLOSE_TARGET,
    DEPTH_GRIPPER_OPEN_TARGET,
)
_IDENTITY_KEYS = ("episode_index", "frame_index", "index")

_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _dataset_column(dataset, key: str) -> np.ndarray:
    hf_dataset = dataset.hf_dataset
    if key not in hf_dataset.column_names:
        raise ValueError(f"Dataset {dataset.root} has no {key!r} column.")
    return hf_dataset.data.column(key).to_numpy(zero_copy_only=False)


def load_depth_gripper_event_targets(dataset) -> dict[str, Tensor]:
    """Load and identity-check the dense label sidecar once per dataset object."""
    try:
        cached = _CACHE.get(dataset)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    root = Path(dataset.root)
    path = root / "meta" / DEPTH_GRIPPER_EVENT_LABEL_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Depth gripper event loss is enabled, but labels are missing: {path}"
        )
    columns = [*_IDENTITY_KEYS, *DEPTH_GRIPPER_EVENT_TARGET_KEYS]
    labels = pd.read_parquet(path, columns=columns)
    if len(labels) != len(dataset):
        raise ValueError(
            f"{path} has {len(labels)} rows, but dataset {root} has {len(dataset)} frames."
        )

    for key in _IDENTITY_KEYS:
        expected = np.asarray(_dataset_column(dataset, key), dtype=np.int64)
        actual = labels[key].to_numpy(dtype=np.int64, copy=False)
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            raise ValueError(
                f"{path} {key!r} disagrees with the dataset at row {mismatch}: "
                f"label={actual[mismatch]}, dataset={expected[mismatch]}."
            )

    result: dict[str, Tensor] = {}
    for key in DEPTH_GRIPPER_EVENT_TARGET_KEYS:
        values = labels[key].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{path} column {key!r} must contain finite values in [0, 1].")
        result[key] = torch.from_numpy(values)
    with suppress(TypeError):
        _CACHE[dataset] = result
    return result
