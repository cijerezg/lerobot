"""Prompt vocabularies, label provenance and clause rendering (plan phase G).

Gate G: prompt snapshots cover every provenance regime, unknown masks are honoured,
the ids are stable, and no source name leaks into the text.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")

from lerobot.datasets.diverse_actor_selection import (  # noqa: E402
    ACTION_LAYOUTS,
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.datasets.diverse_prompt import (  # noqa: E402
    FMB_TASK_TEXT,
    QUALITY_PROVENANCE,
    RETENTION_REASONS,
    UNKNOWN_QUALITY,
    diverse_vocabulary,
    episode_task,
    quality_provenance_id,
    retention_reason_id,
    should_render_quality,
)
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    CONTROL_MODE_TEXT,
    MolmoAct2PackInputsProcessorStep,
    _build_robot_text,
    _build_subtask_generation_text,
)
from lerobot.rl.data_sources.diverse_actor_buffer import (  # noqa: E402
    DiverseActorBuffer,
    DiverseSampleSpec,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "outputs/diverse_robot_dataset_v3"
SOURCE_WORDS = ("droid", "robochallenge", "fmb", "ur7e", "rebot", "arx5", "yam")


@pytest.fixture(scope="module")
def selection():
    if not (DATA_ROOT / "corpus" / "episodes.jsonl").is_file():
        pytest.skip("federated corpora not present")
    return select_actor_anchors(open_federated_corpus(DATA_ROOT))


# ── Stable ids ───────────────────────────────────────────────────────────────


def test_provenance_ids_are_stable_and_unknown_is_zero() -> None:
    assert QUALITY_PROVENANCE[0] == "unknown"
    assert quality_provenance_id(None) == 0
    assert quality_provenance_id("human_reviewed") == 1
    assert quality_provenance_id("human_reviewed_rebot_rubric") == 2
    assert quality_provenance_id("source_derived_automatic") == 3
    assert quality_provenance_id("model_reviewed_rebot_rubric") == 4
    assert RETENTION_REASONS[0] == "unknown"
    assert retention_reason_id("useful_motion") == 1


def test_an_unregistered_provenance_refuses_to_be_folded_into_an_existing_one() -> None:
    with pytest.raises(KeyError, match="never"):
        quality_provenance_id("vibes_based_review")


def test_only_reviewed_quality_renders_by_default() -> None:
    assert should_render_quality("human_reviewed")
    assert should_render_quality("human_reviewed_rebot_rubric")
    assert not should_render_quality("source_derived_automatic")
    assert not should_render_quality("model_reviewed_rebot_rubric")
    assert not should_render_quality(None)
    # ... and the run can still ask for it explicitly.
    assert should_render_quality("source_derived_automatic", render_automatic=True)
    assert should_render_quality("model_reviewed_rebot_rubric", render_automatic=True)


# ── Episode task ─────────────────────────────────────────────────────────────


def test_fmb_task_is_derived_from_its_terminal_primitive() -> None:
    record = {"episode_id": "e", "primitive_intervals": [{"primitive": "grasp"}, {"primitive": "insert"}]}
    text, provenance = episode_task(record, "fmb")
    assert text == FMB_TASK_TEXT
    assert provenance == "derived_from_terminal_primitive"


def test_a_different_fmb_ending_refuses_the_derived_task() -> None:
    record = {"episode_id": "e", "primitive_intervals": [{"primitive": "grasp"}]}
    with pytest.raises(ValueError, match="does not end in"):
        episode_task(record, "fmb")


def test_a_missing_task_is_an_error_not_an_empty_clause() -> None:
    with pytest.raises(ValueError, match="no task string"):
        episode_task({"episode_id": "e"}, "droid")


def test_every_selected_episode_yields_a_task(selection) -> None:
    tasks, subtasks = diverse_vocabulary(selection)
    # v3 corpus minus holdout_episodes.json (18 episodes, 2026-09-19).
    assert len(tasks) == 485
    # The step vocabulary is the reviewed one-action atoms, not the parent intervals
    # (243 of those before 2026-09-09), so it speaks ReBot's grasp/move/release grammar.
    assert len(subtasks) == 862
    assert all(text == text.strip() and text for text in subtasks)
    assert FMB_TASK_TEXT in tasks
    assert tasks == sorted(tasks) and subtasks == sorted(subtasks)


# ── Vocabulary folding ───────────────────────────────────────────────────────


class _FakeMeta:
    def __init__(self) -> None:
        import pandas as pd

        self.tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index(["fold the towel"], name="task"))
        self.subtasks = pd.DataFrame(
            {"subtask_index": [0]}, index=pd.Index(["grasp the towel"], name="subtask")
        )


class _FakeDataset:
    def __init__(self) -> None:
        self.meta = _FakeMeta()


def test_diverse_strings_join_the_master_vocabulary_without_colliding(selection) -> None:
    from lerobot.datasets.diverse_prompt import extend_dataset_vocabulary

    dataset = _FakeDataset()
    tasks, subtasks = extend_dataset_vocabulary(dataset, selection, is_main_process=False)
    # The pre-existing ReBot entry keeps index 0; diverse entries start after it.
    assert min(tasks.values()) >= 1
    assert len(set(tasks.values())) == len(tasks)
    assert len(set(subtasks.values())) == len(subtasks)
    assert int(dataset.meta.tasks["task_index"].max()) == max(tasks.values())


def test_a_string_the_master_already_has_is_reused_not_duplicated(selection) -> None:
    from lerobot.datasets.diverse_prompt import extend_dataset_vocabulary

    dataset = _FakeDataset()
    known = diverse_vocabulary(selection)[0][0]
    import pandas as pd

    dataset.meta.tasks = pd.concat(
        [dataset.meta.tasks, pd.DataFrame({"task_index": [1]}, index=pd.Index([known], name="task"))]
    )
    tasks, _ = extend_dataset_vocabulary(dataset, selection, is_main_process=False)
    assert tasks[known] == 1


# ── Gate G: prompt snapshots per provenance regime ───────────────────────────


def _prompt(metadata) -> str:
    return _build_robot_text(
        task="arrange the fruit in the basket",
        state_string="",
        num_images=2,
        embodiment="Franka Panda",
        current_subtask="grasp the apple",
        metadata=metadata,
    )


def test_human_reviewed_quality_renders() -> None:
    assert " The quality is 4 of 5." in _prompt({"quality": 4, "mistake": False})


def test_automatic_quality_renders_no_quality_clause() -> None:
    """The buffer withholds the integer; the clause must then be absent, not "-1 of 5"."""
    step_metadata = MolmoAct2PackInputsProcessorStep._extract_metadata(
        {
            "metadata_quality": torch.tensor([UNKNOWN_QUALITY]),
            "metadata_mistake": torch.tensor([0.0]),
        },
        1,
    )[0]
    assert "quality" not in step_metadata
    prompt = _prompt(step_metadata)
    assert "quality" not in prompt
    assert "-1" not in prompt
    # The mistake clause still renders: withholding one field does not delete the block.
    assert "The robot made no mistakes." in prompt


def test_the_mistake_clause_tracks_the_anchor_level_flag() -> None:
    assert "The robot made a mistake." in _prompt({"mistake": True})
    assert "The robot made no mistakes." in _prompt({"mistake": False})


def test_embodiment_always_renders_and_the_task_clause_is_never_empty() -> None:
    prompt = _prompt({"mistake": False})
    assert "The robot is a Franka Panda." in prompt
    assert "The task is to arrange the fruit in the basket." in prompt


def test_no_source_name_leaks_into_a_rendered_prompt(selection) -> None:
    from lerobot.datasets.diverse_prompt import extend_dataset_vocabulary

    dataset = _FakeDataset()
    tasks, subtasks = extend_dataset_vocabulary(dataset, selection, is_main_process=False)
    for text in list(tasks) + list(subtasks):
        lowered = text.casefold()
        assert not any(word in lowered for word in SOURCE_WORDS), text


# ── Control-mode clause (diverse v3, 2026-09-19) ─────────────────────────────

EXPECTED_CONTROL_MODES = {
    0: "joint", 1: "joint", 2: "joint", 3: "joint", 4: "joint", 5: "joint", 6: "joint",
    7: "end_effector",  # MolmoAct: xyz + Euler + gripper under the same Franka clause as 0-2
    8: "joint",
}


def test_every_layout_declares_its_control_mode() -> None:
    assert {layout.index: layout.control_mode for layout in ACTION_LAYOUTS} == EXPECTED_CONTROL_MODES
    assert set(EXPECTED_CONTROL_MODES.values()) == set(CONTROL_MODE_TEXT)


def test_the_pack_step_resolves_the_clause_from_the_layout_column() -> None:
    extract = MolmoAct2PackInputsProcessorStep._extract_control_modes
    ids = sorted(EXPECTED_CONTROL_MODES)
    assert extract({"action_layout_id": torch.tensor(ids)}, len(ids)) == [
        EXPECTED_CONTROL_MODES[i] for i in ids
    ]
    # identity_columns() stamps one scalar for the whole batch (rollouts, val_loss, probes).
    assert extract({"action_layout_id": torch.tensor(6)}, 3) == ["joint"] * 3
    assert extract({"action_layout_id": 6}, 2) == ["joint"] * 2
    # No column (a rig-shaped policy) renders no clause.
    assert extract({}, 2) == [None, None]


def test_an_explicit_control_mode_string_wins_and_empty_means_no_clause() -> None:
    extract = MolmoAct2PackInputsProcessorStep._extract_control_modes
    assert extract(
        {"action_layout_id": torch.tensor([7, 7, 7]), "control_mode": ["joint", "", "end_effector"]}, 3
    ) == ["joint", None, "end_effector"]


def _control_mode_prompt(control_mode, embodiment="Franka Panda") -> str:
    return _build_robot_text(
        task="arrange the fruit in the basket",
        state_string="<state_1>",
        num_images=2,
        embodiment=embodiment,
        control_mode=control_mode,
        current_subtask="grasp the apple",
        metadata={"quality": 4, "mistake": False, "speed": 3},
    )


def test_the_clause_follows_the_embodiment_clause() -> None:
    prompt = _control_mode_prompt("end_effector")
    assert (
        "user\nThe robot is a Franka Panda. The control mode is end-effector space. "
        "The task is to arrange the fruit in the basket." in prompt
    )
    assert "The control mode is joint space." in _control_mode_prompt("joint")


def test_a_rebot_row_opens_with_the_control_mode_sentence() -> None:
    """ReBot rows carry no embodiment clause (layout 6 stamps action_layout_id only)."""
    prompt = _control_mode_prompt("joint", embodiment=None)
    assert "The robot is" not in prompt
    assert "<|im_start|>user\nThe control mode is joint space. The task is to arrange" in prompt


def test_none_control_mode_is_the_byte_identical_legacy_prompt() -> None:
    legacy = _build_robot_text(
        task="fold the towel", state_string="<state_1>", num_images=1, embodiment="UR5"
    )
    assert _build_robot_text(
        task="fold the towel", state_string="<state_1>", num_images=1, embodiment="UR5",
        control_mode=None,
    ) == legacy
    assert "control mode" not in legacy
    assert legacy == (
        "<|image|><|im_start|>user\nThe robot is a UR5. The task is to fold the towel. "
        "The current state of the robot is <state_1>. "
        "Given these, what action should the robot take to complete the task?<|im_end|>\n"
        "<|im_start|>assistant\n<action_output>"
    )


def test_the_generation_prompt_carries_the_same_clause() -> None:
    kwargs = {"task": "fold the towel", "state_string": "<state_1>", "num_images": 1}
    with_clause = _build_subtask_generation_text(**kwargs, embodiment="UR5", control_mode="joint")
    assert "user\nThe robot is a UR5. The control mode is joint space. The task is to fold the towel." in with_clause
    rebot = _build_subtask_generation_text(**kwargs, control_mode="joint")
    assert "user\nThe control mode is joint space. The task is to fold the towel." in rebot
    assert "control mode" not in _build_subtask_generation_text(**kwargs, embodiment="UR5")


# ── Buffer columns ───────────────────────────────────────────────────────────


def test_the_buffer_carries_provenance_and_validity(selection) -> None:
    from lerobot.datasets.diverse_prompt import extend_dataset_vocabulary

    dataset = _FakeDataset()
    tasks, subtasks = extend_dataset_vocabulary(dataset, selection, is_main_process=False)
    buffer = DiverseActorBuffer(
        selection,
        DiverseSampleSpec(load_images=False, load_depth=False),
        task_indices=tasks,
        subtask_indices=subtasks,
    )
    picked: dict[str, int] = {}
    for index, row in enumerate(selection.rows):
        picked.setdefault(row["source"], index)
    batch = buffer.collate(sorted(picked.values()))
    info = batch["complementary_info"]
    assert (info["task_index"] >= 0).all()
    assert (info["subtask_index"] >= 0).all()
    assert (info["retention_reason_id"] > 0).all()

    for position, index in enumerate(sorted(picked.values())):
        row = selection.rows[index]
        expected = quality_provenance_id(row["quality_provenance"])
        assert info["quality_provenance_id"][position].item() == expected
        renders = should_render_quality(row["quality_provenance"])
        assert bool(info["metadata_quality_is_valid"][position]) == renders
        if renders:
            assert info["metadata_quality"][position].item() == float(row["quality"])
        else:
            assert info["metadata_quality"][position].item() == UNKNOWN_QUALITY
        # Speed is a computed label with no provenance gate: every anchor carries its atom's.
        assert info["metadata_speed"][position].item() == float(row["speed"])
        assert 1 <= row["speed"] <= 5


def test_speed_renders_as_an_integer_of_five() -> None:
    assert " The speed is 4 of 5." in _prompt({"quality": 4, "mistake": False, "speed": 4})
    step_metadata = MolmoAct2PackInputsProcessorStep._extract_metadata(
        {
            "metadata_quality": torch.tensor([5.0]),
            "metadata_mistake": torch.tensor([0.0]),
            "metadata_speed": torch.tensor([-1.0]),
        },
        1,
    )[0]
    # The -1 sentinel (no atom / no segment) omits the clause, never "-1 of 5".
    assert "speed" not in step_metadata
    assert "-1" not in _prompt(step_metadata)


def test_automatic_quality_can_be_switched_back_on(selection) -> None:
    buffer = DiverseActorBuffer(
        selection,
        DiverseSampleSpec(load_images=False, load_depth=False),
        render_automatic_quality=True,
    )
    index = next(i for i, row in enumerate(selection.rows) if row["source"] == "robochallenge")
    batch = buffer.collate([index])
    assert batch["complementary_info"]["metadata_quality"][0].item() == 5.0
    assert bool(batch["complementary_info"]["metadata_quality_is_valid"][0])


def test_low_quality_and_mistake_examples_are_retained(selection) -> None:
    """Nothing here filters them out; a sampling policy would be a separate decision."""
    qualities = {row["quality"] for row in selection.rows}
    assert min(qualities) == 1
    assert any(row["mistake"] for row in selection.rows)
