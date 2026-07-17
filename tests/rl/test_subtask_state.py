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
    assert shared.subtask_snapshot() == ("reach the cup", 0)

    shared.update_subtask("grasp the cup", 1)
    assert shared.subtask_snapshot() == ("grasp the cup", 1)

    # Snap misses keep the raw text with index -1.
    shared.update_subtask("wiggle mysteriously", -1)
    assert shared.subtask_snapshot() == ("wiggle mysteriously", -1)


def test_clear_subtask_state():
    shared = RTCSharedState()
    shared.update_subtask("reach the cup", 0)
    shared.clear_subtask_state()

    assert shared.subtask_snapshot() == (None, -1)


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

    # Explicit dict beats columns; absent everything = None.
    assert step._extract_metadata({"metadata": {"quality": 5}}, 2) == [{"quality": 5}] * 2
    assert step._extract_metadata({}, 2) == [None, None]
