"""Memory prompt clauses in the MolmoAct2 prompt seam (_build_robot_text)."""

import pytest

pytest.importorskip("transformers", reason="molmoact2 processor module imports policy deps")

from lerobot.policies.molmoact2.processor_molmoact2 import _build_robot_text  # noqa: E402


def build(**overrides):
    kwargs = {
        "task": "fold the towel",
        "discrete_state_string": "",
        "setup_type": "",
        "control_mode": "",
        "add_setup_tokens": False,
        "add_control_tokens": False,
        "num_images": 0,
    }
    kwargs.update(overrides)
    return _build_robot_text(**kwargs)


def test_legacy_prompt_is_byte_identical_when_memory_off():
    """All memory params None → the exact pre-memory prompt (checkpoint compatibility)."""
    prompt = build()
    assert (
        "The task is to fold the towel. The setup is" in prompt
    ), f"unexpected legacy prompt: {prompt}"
    assert "Steps already completed" not in prompt
    assert "current step" not in prompt
    assert "quality" not in prompt


def test_current_subtask_clause():
    prompt = build(current_subtask="grab the far side")
    assert "The current step is grab the far side." in prompt


def test_metadata_clause_partial_rendering():
    prompt = build(metadata={"quality": 5, "mistake": False, "speed": "fast"})
    assert "The quality is 5 of 5." in prompt
    assert "The robot made no mistakes." in prompt
    assert "The speed is fast." in prompt

    prompt = build(metadata={"mistake": True})
    assert "The robot made a mistake." in prompt
    assert "quality" not in prompt

    prompt = build(metadata={})
    assert "quality" not in prompt and "mistake" not in prompt and "speed" not in prompt


def test_clause_order_task_then_memory_then_setup():
    prompt = build(current_subtask="step two")
    task_pos = prompt.index("The task is to")
    current_pos = prompt.index("The current step is")
    setup_pos = prompt.index("The setup is")
    assert task_pos < current_pos < setup_pos


# ── Phase 3: subtask generation prompt + vocab snap ──────────────────────────

from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    _build_subtask_generation_text,
    snap_to_subtask_vocab,
)

VOCAB = ["reach the cup", "grasp the cup", "lift the cup"]


def test_generation_prompt_asks_for_next_step():
    prompt = _build_subtask_generation_text(
        task="put the cup on the shelf",
        discrete_state_string="",
        setup_type="",
        add_setup_tokens=False,
        num_images=0,
    )
    assert "what step should the robot perform next?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
    # The action-output token never appears — this prompt decodes text, not actions.
    assert "<action" not in prompt


def test_snap_to_subtask_vocab():
    assert snap_to_subtask_vocab("grasp the cup", VOCAB) == 1
    assert snap_to_subtask_vocab("  Grasp the Cup. ", VOCAB) == 1  # case/punct-insensitive
    assert snap_to_subtask_vocab("grasp cup", VOCAB) == 1          # fuzzy
    assert snap_to_subtask_vocab("do a backflip", VOCAB) == -1     # no match
