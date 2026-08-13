from __future__ import annotations

import pandas as pd
import pytest
import torch
from datasets import Dataset

from lerobot.utils.depth_gripper_events import load_depth_gripper_event_targets


class _Dataset:
    def __init__(self, root, *, frame_indices=(0, 1)) -> None:
        self.root = root
        self.hf_dataset = Dataset.from_dict(
            {
                "episode_index": [0, 0],
                "frame_index": list(frame_indices),
                "index": [0, 1],
            }
        )

    def __len__(self) -> int:
        return len(self.hf_dataset)


def _write_labels(root, *, frame_indices=(0, 1)) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": list(frame_indices),
            "index": [0, 1],
            "depth_gripper_close_target": [0.0, 1.0],
            "depth_gripper_open_target": [0.5, 0.25],
        }
    ).to_parquet(meta / "depth_gripper_event_labels.parquet", index=False)


def test_sidecar_loader_checks_identities_and_returns_float32(tmp_path) -> None:
    _write_labels(tmp_path)

    targets = load_depth_gripper_event_targets(_Dataset(tmp_path))

    torch.testing.assert_close(targets["depth_gripper_close_target"], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(targets["depth_gripper_open_target"], torch.tensor([0.5, 0.25]))
    assert all(value.dtype == torch.float32 for value in targets.values())


def test_sidecar_loader_rejects_identity_mismatch(tmp_path) -> None:
    _write_labels(tmp_path, frame_indices=(0, 2))

    with pytest.raises(ValueError, match="frame_index.*row 1"):
        load_depth_gripper_event_targets(_Dataset(tmp_path))
