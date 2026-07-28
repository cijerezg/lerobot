from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from datasets import Dataset

from lerobot.configs.default import DatasetConfig, OfflineDatasetSourceConfig
from lerobot.rl.buffer import concatenate_batch_transitions
from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
from lerobot.rl.offline_dataset_utils import (
    _weighted_batch_sizes,
    get_offline_dataset_sources,
    materialize_dataset_labels,
    resolve_task_strings,
)


def _fake_dataset(tasks, task_indices, subtasks, subtask_indices, root):
    return SimpleNamespace(
        root=root,
        meta=SimpleNamespace(
            tasks=pd.DataFrame(
                {"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task")
            ),
            subtasks=pd.DataFrame(
                {"subtask_index": range(len(subtasks))}, index=pd.Index(subtasks, name="subtask")
            ),
        ),
        hf_dataset=Dataset.from_dict(
            {"task_index": task_indices, "subtask_index": subtask_indices}
        ),
    )


def _fake_buffer(capacity):
    return SimpleNamespace(
        capacity=capacity,
        storage_device=torch.device("cpu"),
        complementary_info={},
        complementary_info_keys=[],
        has_complementary_info=False,
    )


def test_collection_orders_normalization_source_first_and_preserves_weights():
    cfg = SimpleNamespace(
        dataset=DatasetConfig(
            repo_id="local/tasks",
            sources=[
                OfflineDatasetSourceConfig(root="/data/pick", name="pick", weight=1),
                OfflineDatasetSourceConfig(
                    root="/data/composed",
                    name="composed",
                    weight=2,
                    normalization_source=True,
                ),
            ],
        )
    )

    sources = get_offline_dataset_sources(cfg)

    assert [source.name for source in sources] == ["composed", "pick"]
    assert [source.weight for source in sources] == [2.0, 1.0]


def test_weighted_batch_sizes_are_exact_and_keep_every_source_present():
    assert _weighted_batch_sizes(96, [1, 1, 2]) == [24, 24, 48]
    with pytest.raises(ValueError, match="at least the number"):
        _weighted_batch_sizes(2, [1, 1, 1])


def test_current_task_and_subtask_labels_overlay_cached_buffer():
    vocabulary = _fake_dataset(
        ["pick object"], [0], ["grasp object"], [0], "/main"
    )
    source = _fake_dataset(
        ["pick and place"],
        [0, 0, 0],
        ["Grasp object.", "place object"],
        [0, 1, 1],
        "/extra",
    )
    buffer = _fake_buffer(capacity=3)

    materialize_dataset_labels(buffer, source, vocabulary, source_index=1)

    assert vocabulary.meta.tasks.loc["pick and place", "task_index"] == 1
    assert buffer.complementary_info["task_index"].tolist() == [1, 1, 1]
    assert buffer.complementary_info["subtask_index"].tolist() == [0, 1, 1]
    assert buffer.complementary_info["source_index"].tolist() == [1, 1, 1]
    raw = {"complementary_info": {"task_index": buffer.complementary_info["task_index"]}}
    assert resolve_task_strings(raw, vocabulary, "fallback", 3) == ["pick and place"] * 3


def test_missing_task_index_uses_rollout_fallback():
    dataset = _fake_dataset(["pick"], [0], ["grasp"], [0], "/main")
    raw = {"complementary_info": {"task_index": torch.tensor([-1, 0])}}
    assert resolve_task_strings(raw, dataset, "rollout task", 2) == ["rollout task", "pick"]


def test_molmoact2_resolves_a_different_task_for_each_sample():
    trainer = object.__new__(MolmoAct2Trainer)
    trainer._task_index_to_name = {0: "pick", 1: "pick and place"}
    raw = {"complementary_info": {"task_index": torch.tensor([1, -1, 0])}}

    assert trainer._resolve_batch_tasks(raw, "rollout fallback", 3) == [
        "pick and place", "rollout fallback", "pick"
    ]


def _transition_batch(task_index=None):
    batch = {
        "state": {"observation.state": torch.zeros(1, 1)},
        "action": torch.zeros(1, 1, 1),
        "reward": torch.zeros(1),
        "next_state": {"observation.state": torch.zeros(1, 1)},
        "done": torch.zeros(1, dtype=torch.bool),
        "truncated": torch.zeros(1, dtype=torch.bool),
    }
    if task_index is not None:
        batch["complementary_info"] = {"task_index": torch.tensor([task_index])}
    return batch


def test_missing_collection_indices_are_padded_with_negative_one():
    online = _transition_batch()
    offline = _transition_batch(task_index=2)

    combined = concatenate_batch_transitions(online, offline, action_dim=1)

    assert combined["complementary_info"]["task_index"].tolist() == [-1, 2]
