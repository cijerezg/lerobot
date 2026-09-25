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


def test_subtask_script_walk():
    shared = RTCSharedState()
    shared.set_subtask_script([("grasp the cup", 1), ("return to home", 5), ("wiggle", -1)])
    assert shared.subtask_snapshot() == ("grasp the cup", 1)

    assert shared.advance_subtask(1) == 1
    assert shared.advance_subtask(1) == 2
    assert shared.subtask_snapshot() == ("wiggle", -1)
    # Clamped at the end, then back one.
    assert shared.advance_subtask(1) == 2
    assert shared.advance_subtask(-1) == 1
    assert shared.subtask_snapshot() == ("return to home", 5)

    # Episode reset rewinds to the first entry; back clamps at the start.
    shared.clear_subtask_state()
    assert shared.subtask_snapshot() == ("grasp the cup", 1)
    assert shared.advance_subtask(-1) == 0


def test_subtask_console_home_override():
    from types import SimpleNamespace

    from lerobot.rl.subtask_console import SubtaskConsole

    shared = RTCSharedState()
    console = SubtaskConsole(["grasp the cup", "move the cup"], "return to home", ["grasp the cup", "return to home"], shared)
    assert console.home == ("return to home", 1, {})
    assert shared.subtask_snapshot() == ("grasp the cup", 0)

    # r latches home immediately without moving the cursor; the next n resumes the script.
    console._on_press(SimpleNamespace(char="r"))
    assert shared.subtask_snapshot() == ("return to home", 1)
    assert shared.subtask_cursor == 0
    console._on_press(SimpleNamespace(char="n"))
    assert shared.subtask_snapshot() == ("move the cup", -1)
    console._on_press(SimpleNamespace(char="r"))
    console._on_press(SimpleNamespace(char="b"))
    assert shared.subtask_snapshot() == ("grasp the cup", 0)
    # No precision/contact lists: nothing latched, the prompt metadata stays the constants.
    assert shared.subtask_metadata_snapshot() == {}


def test_subtask_console_latches_precision_and_contact():
    from types import SimpleNamespace

    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig
    from lerobot.rl.rtc_actor_runtime import _merge_subtask_metadata
    from lerobot.rl.subtask_console import SubtaskConsole, script_metadata

    cfg = MolmoAct2RLConfig(
        eval_subtasks=["grasp the cup", "place the cup"],
        eval_subtask_precisions=[4, 2],
        eval_subtask_contacts=["side-pinch", "set-down"],
    )
    metadata, home_metadata = script_metadata(cfg)
    assert metadata == [{"precision": 4, "contact": 1}, {"precision": 2, "contact": 9}]
    assert home_metadata == {"precision": 1, "contact": 14}

    shared = RTCSharedState()
    console = SubtaskConsole(
        cfg.eval_subtasks, cfg.eval_home_subtask, [], shared, metadata=metadata, home_metadata=home_metadata
    )
    constants = {"quality": 5, "mistake": False, "speed": 5}
    assert _merge_subtask_metadata(constants, shared) == {**constants, "precision": 4, "contact": 1}

    console._on_press(SimpleNamespace(char="n"))
    assert shared.subtask_snapshot() == ("place the cup", -1)
    assert shared.subtask_metadata_snapshot() == {"precision": 2, "contact": 9}
    console._on_press(SimpleNamespace(char="r"))
    assert shared.subtask_snapshot() == ("return to home", -1)
    assert shared.subtask_metadata_snapshot() == {"precision": 1, "contact": 14}
    console._on_press(SimpleNamespace(char="b"))
    assert shared.subtask_metadata_snapshot() == {"precision": 4, "contact": 1}
    # Episode reset rewinds metadata with the step; metadata off stays None.
    console._on_press(SimpleNamespace(char="n"))
    shared.clear_subtask_state()
    assert shared.subtask_metadata_snapshot() == {"precision": 4, "contact": 1}
    assert _merge_subtask_metadata(None, shared) is None


def test_script_metadata_per_channel():
    """One list set = that channel only; neither set = no per-step keys (today's prompt)."""
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig
    from lerobot.rl.subtask_console import script_metadata

    only_precision = MolmoAct2RLConfig(eval_subtasks=["grasp the cup"], eval_subtask_precisions=[3])
    assert script_metadata(only_precision) == ([{"precision": 3}], {"precision": 1})
    neither = MolmoAct2RLConfig(eval_subtasks=["grasp the cup"])
    assert script_metadata(neither) == ([{}], {})


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

    # Precision/contact columns: -1 = absent, the key is omitted.
    out = step._extract_metadata(
        {
            "metadata_quality": torch.tensor([5.0, 5.0]),
            "metadata_mistake": torch.tensor([0.0, 0.0]),
            "metadata_speed": torch.tensor([3.0, 3.0]),
            "metadata_precision": torch.tensor([4.0, -1.0]),
            "metadata_contact": torch.tensor([14.0, -1.0]),
        },
        batch_size=2,
    )
    assert out[0] == {"quality": 5, "mistake": False, "speed": 3, "precision": 4, "contact": 14}
    assert out[1] == {"quality": 5, "mistake": False, "speed": 3}

    # Explicit dict beats columns; absent everything = None.
    assert step._extract_metadata({"metadata": {"quality": 5}}, 2) == [{"quality": 5}] * 2
    assert step._extract_metadata({}, 2) == [None, None]
