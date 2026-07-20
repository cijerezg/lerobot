"""Rollout-side subtask bookkeeping (Phase 3 slice 2)."""

import pytest

pytest.importorskip("transformers", reason="runtime module imports policy deps")

from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2PackInputsProcessorStep,
)
from lerobot.rl.rtc_actor_runtime import RTCSharedState  # noqa: E402


def test_update_subtask():
    shared = RTCSharedState()

    shared.update_subtask("reach the cup", 0)
    assert shared.subtask_snapshot() == ("reach the cup", 0, "")

    shared.update_subtask("grasp the cup", 1, "I reached the cup.")
    assert shared.subtask_snapshot() == ("grasp the cup", 1, "I reached the cup.")

    # summary=None (decode without a memory span) keeps the current memory.
    shared.update_subtask("lift the cup", 2)
    assert shared.subtask_snapshot() == ("lift the cup", 2, "I reached the cup.")

    # Snap misses keep the raw text with index -1.
    shared.update_subtask("wiggle mysteriously", -1)
    assert shared.subtask_snapshot()[:2] == ("wiggle mysteriously", -1)


def test_clear_subtask_state():
    shared = RTCSharedState()
    shared.update_subtask("reach the cup", 0, "I reached the cup.")
    shared.clear_subtask_state()

    assert shared.subtask_snapshot() == (None, -1, "")


def test_extract_summaries():
    import torch

    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.summary_texts = ["I did A.", "I did A and B."]

    # Offline path: index columns rendered through the table; -1 = empty memory.
    prev, target = step._extract_summaries(
        {
            "summary_prev_index": torch.tensor([-1, 0]),
            "summary_target_index": torch.tensor([0, 1]),
        },
        batch_size=2,
    )
    assert prev == ["", "I did A."]
    assert target == ["I did A.", "I did A and B."]

    # Rollout path: conditioning strings only, no target.
    prev, target = step._extract_summaries({"summary": ["I did A."]}, batch_size=1)
    assert prev == ["I did A."] and target == [None]

    # Not wired: all None (subtask-only training).
    assert step._extract_summaries({}, batch_size=2) == ([None, None], [None, None])
    step.summary_texts = []
    prev, target = step._extract_summaries({"summary_prev_index": torch.tensor([0])}, batch_size=1)
    assert prev == [None] and target == [None]


def test_extract_metadata_from_columns():
    import torch

    step = object.__new__(MolmoAct2PackInputsProcessorStep)

    out = step._extract_metadata(
        {
            "metadata_quality": torch.tensor([5.0, 3.0]),
            "metadata_mistake": torch.tensor([0.0, 1.0]),
            "metadata_speed": torch.tensor([2.0, 0.0]),
        },
        batch_size=2,
    )
    assert out[0] == {"quality": 5, "mistake": False, "speed": 2}
    assert out[1] == {"quality": 3, "mistake": True, "speed": 0}

    # Speed column omitted (metadata_annotate.py datasets): clause renders partially.
    out = step._extract_metadata(
        {
            "metadata_quality": torch.tensor([4.0]),
            "metadata_mistake": torch.tensor([1.0]),
        },
        batch_size=1,
    )
    assert out[0] == {"quality": 4, "mistake": True}

    # Explicit dict beats columns; absent everything = None.
    assert step._extract_metadata({"metadata": {"quality": 5}}, 2) == [{"quality": 5}] * 2
    assert step._extract_metadata({}, 2) == [None, None]
