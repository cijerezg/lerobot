"""Control-mode clause through the real pack step, the probe adapter and a val_loss batch.

The clause is read from the layout record behind ``action_layout_id`` (diverse v3,
2026-09-19), so the checks here are about the plumbing: the column every buffer stamps
reaches the prompt, the scalar identity column expands to the batch, the adapter's
per-row override wins, and a batch without the column is the legacy prompt.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")

from lerobot.datasets.contact_vocab import CONTACT_VOCAB  # noqa: E402
from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    MolmoAct2PackInputsProcessorStep,
    _build_robot_text,
)
from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter  # noqa: E402
from lerobot.probes.utils import identity_columns  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.scripts.diverse_smoke import (  # noqa: E402
    _SPEED_CLAUSE,
    parse_contact_clauses,
    parse_precision_clauses,
)
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

BASE_PATH = Path(__file__).resolve().parents[3] / "outputs/MolmoAct2"
ROLE_KEYS = [f"{OBS_IMAGES}.external_0", f"{OBS_IMAGES}.wrist_0"]
TASK = "put the sock in the basket"


@pytest.fixture(scope="module")
def pack_step():
    if not (BASE_PATH / "processor_config.json").is_file():
        pytest.skip("MolmoAct2 base checkpoint not present")
    return MolmoAct2PackInputsProcessorStep(
        base_path=str(BASE_PATH),
        action_mode="continuous",
        image_keys=list(ROLE_KEYS),
        chunk_size=30,
        max_action_dim=8,
    )


def _observation(batch_size: int) -> dict:
    observation = {OBS_STATE: torch.zeros(batch_size, 8), "task": [TASK] * batch_size}
    for key in ROLE_KEYS:
        observation[key] = torch.zeros(batch_size, 3, 64, 64, dtype=torch.uint8)
    return observation


def _opening(pack_step, ids) -> str:
    """The decoded prompt of one row from "<|im_start|>user" up to the task sentence's end."""
    text = pack_step.processor.tokenizer.decode(ids, skip_special_tokens=False)
    start = text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    return text[start : text.index(f"{TASK}.") + len(f"{TASK}.")]


def _user_turns(pack_step, complementary: dict, batch_size: int) -> list[str]:
    out = pack_step(
        create_transition(
            observation=_observation(batch_size),
            action=torch.zeros(batch_size, 30, 8),
            complementary_data=complementary,
        )
    )
    return [_opening(pack_step, ids) for ids in out[TransitionKey.COMPLEMENTARY_DATA]["input_ids"]]


# ── Real tokenizer step ──────────────────────────────────────────────────────


def test_a_per_row_layout_column_renders_each_row_its_own_clause(pack_step) -> None:
    turns = _user_turns(
        pack_step,
        {"action_layout_id": torch.tensor([7, 0, 6]), "embodiment_index": torch.tensor([0, 0, -1])},
        3,
    )
    assert turns == [
        f"The robot is a Franka Panda. The control mode is end-effector space. The task is to {TASK}.",
        f"The robot is a Franka Panda. The control mode is joint space. The task is to {TASK}.",
        f"The control mode is joint space. The task is to {TASK}.",
    ]


def test_a_rebot_training_row_opens_with_the_control_mode_sentence(pack_step) -> None:
    """RoleAlignedBuffer stamps action_layout_id (layout 6) and no embodiment index."""
    turns = _user_turns(pack_step, {"action_layout_id": torch.full((2,), 6)}, 2)
    assert turns == [f"The control mode is joint space. The task is to {TASK}."] * 2


def test_a_batch_without_the_layout_column_is_the_legacy_prompt(pack_step) -> None:
    assert _user_turns(pack_step, {}, 2) == [f"The task is to {TASK}."] * 2


def test_the_explicit_string_wins_over_the_layout_column(pack_step) -> None:
    turns = _user_turns(
        pack_step, {"action_layout_id": torch.tensor([7, 7]), "control_mode": ["", "joint"]}, 2
    )
    assert turns == [f"The task is to {TASK}.", f"The control mode is joint space. The task is to {TASK}."]


def test_the_generation_prompt_renders_the_clause_too(pack_step) -> None:
    out = pack_step._pack_subtask_generation(
        create_transition(
            observation=_observation(1), complementary_data={"action_layout_id": torch.tensor(7)}
        )
    )
    text = pack_step.processor.tokenizer.decode(
        out[TransitionKey.COMPLEMENTARY_DATA]["input_ids"][0], skip_special_tokens=False
    )
    assert f"user\nThe control mode is end-effector space. The task is to {TASK}." in text
    assert "what step should the robot perform next?" in text


# ── val_loss-style batch: the scalar identity column ─────────────────────────


def _role_shaped_cfg():
    return SimpleNamespace(
        policy=SimpleNamespace(input_features={key: None for key in ROLE_KEYS}, image_keys=list(ROLE_KEYS)),
        diverse=SimpleNamespace(rebot_layout="rebot_b601_joint7_commanded"),
    )


def test_a_val_loss_batch_carries_the_clause_on_every_row(pack_step) -> None:
    """val_loss._pack puts identity_columns(cfg) -- one scalar action_layout_id -- beside
    per-frame subtask and metadata lists; the clause must land on every frame."""
    identity = identity_columns(_role_shaped_cfg())
    assert identity == {"action_layout_id": 6}
    complementary = {
        **identity,
        "subtask": ["grasp the sock", "move the sock to the basket", "release the sock"],
        "metadata": [{"quality": 5, "mistake": False, "speed": 5}] * 3,
    }
    turns = _user_turns(pack_step, complementary, 3)
    assert turns == [f"The control mode is joint space. The task is to {TASK}."] * 3


# ── Probe adapter override ───────────────────────────────────────────────────


def _adapter(pack_step) -> MolmoAct2Adapter:
    return MolmoAct2Adapter(
        policy=None,
        preprocessor=PolicyProcessorPipeline(steps=[pack_step]),
        postprocessor=None,
        device=torch.device("cpu"),
        cfg=_role_shaped_cfg(),
    )


def _decode_rows(pack_step, batch: dict) -> list[str]:
    return [_opening(pack_step, ids) for ids in batch["input_ids"]]


def test_the_adapter_override_sets_the_clause_per_row(pack_step) -> None:
    adapter = _adapter(pack_step)
    obs = {key: value[:1] for key, value in _observation(1).items() if key != "task"}
    batch = adapter._make_batch_multi(
        obs, TASK, ["grasp the sock"] * 3, control_modes=["end_effector", "", "joint"]
    )
    assert _decode_rows(pack_step, batch) == [
        f"The control mode is end-effector space. The task is to {TASK}.",
        f"The task is to {TASK}.",
        f"The control mode is joint space. The task is to {TASK}.",
    ]


def test_without_the_override_the_adapter_renders_the_identity_layout(pack_step) -> None:
    adapter = _adapter(pack_step)
    obs = {key: value[:1] for key, value in _observation(1).items() if key != "task"}
    batch = adapter._make_batch_multi(obs, TASK, ["grasp the sock"] * 2, embodiments=["", "UR5"])
    assert _decode_rows(pack_step, batch) == [
        f"The control mode is joint space. The task is to {TASK}.",
        f"The robot is a UR5. The control mode is joint space. The task is to {TASK}.",
    ]


def test_the_override_length_must_match_the_rows(pack_step) -> None:
    adapter = _adapter(pack_step)
    obs = {key: value[:1] for key, value in _observation(1).items() if key != "task"}
    with pytest.raises(ValueError, match="control_modes must have length 2"):
        adapter._make_batch_multi(obs, TASK, ["grasp the sock"] * 2, control_modes=["joint"])


def test_the_prompt_parser_gives_the_clause_its_own_group(pack_step) -> None:
    """_prompt_token_groups finds each clause by its marker; the control-mode sentence
    must resolve whether it follows the embodiment clause or opens the prompt."""
    tokenizer = pack_step.processor.tokenizer
    find = MolmoAct2Adapter._find_subsequence
    for complementary, expected in (
        ({"action_layout_id": torch.tensor([7]), "embodiment_index": torch.tensor([0])},
         ["embodiment", "control_mode", "task", "state", "question"]),
        ({"action_layout_id": torch.tensor([6])}, ["control_mode_first", "task", "state", "question"]),
        ({}, ["task_first", "state", "question"]),
    ):
        out = pack_step(
            create_transition(
                observation=_observation(1), action=torch.zeros(1, 30, 8), complementary_data=complementary
            )
        )
        row = out[TransitionKey.COMPLEMENTARY_DATA]["input_ids"][0].tolist()
        starts = {}
        for name, marker in MolmoAct2Adapter._PROMPT_CLAUSES:
            pos = find(row, tokenizer.encode(marker, add_special_tokens=False))
            if pos is not None:
                starts[name] = pos
        assert [name for name, _ in sorted(starts.items(), key=lambda kv: kv[1])] == expected


def test_the_action_key_is_untouched_by_the_clause(pack_step) -> None:
    """The clause is text only: the packed action and its width mask do not change."""
    transition = create_transition(
        observation=_observation(1),
        action=torch.ones(1, 30, 7),
        complementary_data={"action_layout_id": torch.tensor([7])},
    )
    with_clause = pack_step(transition)
    without = pack_step(create_transition(observation=_observation(1), action=torch.ones(1, 30, 7)))
    assert torch.equal(with_clause[TransitionKey.ACTION], without[TransitionKey.ACTION])
    assert torch.equal(
        with_clause[TransitionKey.COMPLEMENTARY_DATA]["action_dim_is_pad"],
        without[TransitionKey.COMPLEMENTARY_DATA]["action_dim_is_pad"],
    )


# ── Smoke clause parser: precision and contact ───────────────────────────────


def _metadata_prompt(**metadata) -> str:
    return _build_robot_text(
        task=TASK,
        state_string="",
        num_images=0,
        current_subtask="grasp the sock",
        metadata={"quality": 5, "mistake": False, "speed": 3, **metadata},
    )


@pytest.mark.parametrize("level", range(1, 6))
def test_the_smoke_parser_round_trips_every_precision_level(level) -> None:
    text = _metadata_prompt(precision=level, contact=0)
    assert parse_precision_clauses(text) == [level]
    assert parse_contact_clauses(text) == [0]
    assert _SPEED_CLAUSE.findall(text) == ["3"]


@pytest.mark.parametrize("element", CONTACT_VOCAB, ids=lambda e: e.slug)
def test_the_smoke_parser_round_trips_every_contact_phrase(element) -> None:
    text = _metadata_prompt(precision=2, contact=element.code)
    assert f" The contact is {element.phrase}. Given these" in text
    assert parse_contact_clauses(text) == [element.code]
    assert parse_precision_clauses(text) == [2]


def test_a_prompt_without_the_new_clauses_parses_to_none() -> None:
    text = _metadata_prompt()
    assert "precision" not in text and "contact" not in text
    assert parse_precision_clauses(text) == []
    assert parse_contact_clauses(text) == []
    assert parse_precision_clauses(_build_robot_text(task=TASK, state_string="", num_images=0)) == []


def test_a_contact_phrase_outside_the_vocabulary_is_an_error() -> None:
    with pytest.raises(ValueError, match="not in the vocabulary"):
        parse_contact_clauses("The speed is 3 of 5. The contact is a firm handshake. Given these")


def test_the_pack_step_columns_render_clauses_the_smoke_parser_reads_back(pack_step) -> None:
    """Through the real tokenizer: the buffer columns (-1 = unlabelled) come back out of
    the decoded prompt exactly, which is the check diverse_smoke runs on every sample."""
    precision = [1, 5, -1, 3]
    contact = [0, 14, 7, -1]
    out = pack_step(
        create_transition(
            observation=_observation(4),
            action=torch.zeros(4, 30, 8),
            complementary_data={
                "metadata_quality": torch.full((4,), 5.0),
                "metadata_quality_is_valid": torch.ones(4, dtype=torch.bool),
                "metadata_mistake": torch.zeros(4),
                "metadata_speed": torch.full((4,), 4.0),
                "metadata_precision": torch.tensor(precision, dtype=torch.bfloat16),
                "metadata_contact": torch.tensor(contact, dtype=torch.bfloat16),
            },
        )
    )
    tokenizer = pack_step.processor.tokenizer
    for i, ids in enumerate(out[TransitionKey.COMPLEMENTARY_DATA]["input_ids"]):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        assert parse_precision_clauses(text) == ([precision[i]] if precision[i] >= 0 else [])
        assert parse_contact_clauses(text) == ([contact[i]] if contact[i] >= 0 else [])
        assert _SPEED_CLAUSE.findall(text) == ["4"]
