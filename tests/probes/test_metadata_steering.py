"""Metadata steering: the steering conditions, the mistake-window sampler, the adapter's
clause markers and the trainer's forwarded keys."""

from types import SimpleNamespace

import pytest
import torch

from lerobot.probes import metadata_steering as ms
from lerobot.probes.utils import sample_mistake_windows
from lerobot.rl.molmoact2.rl_molmoact2_trainer import _forwarded_complementary_keys


def test_conditions_carry_quality_mistake_and_speed_only() -> None:
    """The steering rows render today's prompts: no precision or contact key is set."""
    for metadata in ms._STEERED.values():
        assert set(metadata) == {"quality", "mistake", "speed"}


class _FakeDataset:
    """Two episodes of 40 frames; ``frame_index`` restarts per episode."""

    def __init__(self) -> None:
        self.hf_dataset = [
            {"episode_index": torch.tensor(ep), "frame_index": torch.tensor(fr)}
            for ep in (0, 1) for fr in range(40)
        ]

    def __len__(self) -> int:
        return len(self.hf_dataset)


def test_mistake_windows_sample_inside_every_window_on_the_stride_grid() -> None:
    dataset = _FakeDataset()
    # episode 0: frames 10..21 flagged; episode 1: frames 45..47 and 60..71 flagged
    flagged = set(range(10, 22)) | set(range(45, 48)) | set(range(60, 72))
    gt = {idx: {"mistake": idx in flagged} for idx in range(len(dataset))}

    samples = sample_mistake_windows(dataset, gt, 3, [0, 1], stride=3, taken={21})

    global_idx = [s[2] for s in samples]
    assert all(idx in flagged for idx in global_idx)
    assert all(dataset.hf_dataset[idx]["frame_index"].item() % 3 == 0 for idx in global_idx)
    assert 21 not in global_idx  # already in the even sample
    assert len(set(global_idx)) == len(global_idx)
    # 10..21: on-grid 12, 15, 18, 21 -> evenly 12, 15, 21; 21 is taken
    assert [idx for idx in global_idx if idx < 40] == [12, 15]
    # 45..47 is frames 5..7: one on-grid frame (6 = global 46), sampled once
    assert [idx for idx in global_idx if 40 <= idx < 50] == [46]
    # 60..71 is frames 20..31: on-grid 21, 24, 27, 30 -> evenly 21, 24, 30
    assert [idx for idx in global_idx if idx >= 60] == [61, 64, 70]


def test_no_labels_means_no_mistake_frames() -> None:
    assert sample_mistake_windows(_FakeDataset(), {}, 4, [0, 1], stride=3, taken=set()) == []


def test_prompt_markers_find_the_new_clauses_in_order() -> None:
    """The adapter's clause markers, run over the processor's own rendered prompt."""
    pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")
    from lerobot.policies.molmoact2.processor_molmoact2 import _build_robot_text
    from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter

    text = _build_robot_text(
        task="put the sock in the basket",
        discrete_state_string="<state>",
        num_images=0,
        metadata={"quality": 5, "mistake": False, "speed": 4, "precision": 2, "contact": 0},
    )
    chars = list(text)
    starts = {}
    for name, marker in MolmoAct2Adapter._PROMPT_CLAUSES:
        pos = MolmoAct2Adapter._find_subsequence(chars, list(marker))
        if pos is not None:
            starts[name] = pos
    assert [name for name, _ in sorted(starts.items(), key=lambda kv: kv[1])] == [
        "task_first", "state", "metadata_quality", "metadata_mistake", "metadata_speed",
        "metadata_precision", "metadata_contact", "question",
    ]


def test_trainer_forwards_the_new_columns_only_when_present() -> None:
    cfg = SimpleNamespace(policy=SimpleNamespace(future_visual_loss=None))
    with_columns = {
        "metadata_speed": torch.tensor([3.0]),
        "metadata_precision": torch.tensor([2.0]),
        "metadata_contact": torch.tensor([14.0]),
        "episode_id": torch.tensor([0]),
    }
    keys = _forwarded_complementary_keys(with_columns, cfg)
    assert {"metadata_speed", "metadata_precision", "metadata_contact"} <= set(keys)
    assert "episode_id" not in keys
    assert _forwarded_complementary_keys({"metadata_speed": torch.tensor([3.0])}, cfg) == ["metadata_speed"]
