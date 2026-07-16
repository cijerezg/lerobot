"""Rollout-side subtask/done-list bookkeeping (Phase 3 slice 2)."""

import pytest

pytest.importorskip("transformers", reason="runtime module imports policy deps")

from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2PackInputsProcessorStep,
)
from lerobot.rl.rtc_actor_runtime import RTCSharedState  # noqa: E402


def test_update_subtask_append_on_switch():
    shared = RTCSharedState()

    shared.update_subtask("reach the cup", 0)
    name, index, done_names, done_ids = shared.subtask_snapshot()
    assert (name, index) == ("reach the cup", 0)
    assert done_names == [] and done_ids == []

    # Same subtask again: no append.
    shared.update_subtask("reach the cup", 0)
    assert shared.subtask_snapshot()[2] == []

    # Switch: previous one is done.
    shared.update_subtask("grasp the cup", 1)
    name, index, done_names, done_ids = shared.subtask_snapshot()
    assert (name, index) == ("grasp the cup", 1)
    assert done_names == ["reach the cup"]
    assert done_ids == [0]


def test_update_subtask_unsnapped_names_tracked_ids_not():
    shared = RTCSharedState()
    shared.update_subtask("wiggle mysteriously", -1)  # snap missed the vocab
    shared.update_subtask("grasp the cup", 1)

    _, _, done_names, done_ids = shared.subtask_snapshot()
    assert done_names == ["wiggle mysteriously"]  # prompt still sees it
    assert done_ids == []  # ids only carry vocabulary entries


def test_clear_subtask_state():
    shared = RTCSharedState()
    shared.update_subtask("reach the cup", 0)
    shared.update_subtask("grasp the cup", 1)
    shared.clear_subtask_state()

    name, index, done_names, done_ids = shared.subtask_snapshot()
    assert name is None and index == -1
    assert done_names == [] and done_ids == []


def test_extract_done_lists_prefers_rollout_strings():
    # Method under test touches no processor state — skip the heavy __post_init__.
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.subtask_names = ["reach the cup", "grasp the cup"]

    out = step._extract_done_lists({"done_list": [["reach the cup"]]}, batch_size=1)
    assert out == [["reach the cup"]]

    out = step._extract_done_lists({"done_list": [[]]}, batch_size=1)
    assert out == [[]]  # episode start: memory on, nothing done

    out = step._extract_done_lists({}, batch_size=1)
    assert out == [None]  # feature off


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
