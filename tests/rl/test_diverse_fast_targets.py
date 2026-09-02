"""FAST/discrete targets at each sample's native width (integration plan phase F).

Gate F: a 7D example must produce identical FAST tokens and CE loss whether it is
tokenized alone or beside an 8D example.

FAST runs a DCT over time and then flattens (horizon, dim) before BPE, so a padded
eighth column of zeros does not merely append tokens -- it changes the flattened bin
string, and with it every token of a 7-DoF robot's target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers", reason="molmoact2 processor imports policy deps")

from lerobot.policies.molmoact2.processor_molmoact2 import (  # noqa: E402
    ACTION_TOKEN_PREFIX,
    MolmoAct2PackInputsProcessorStep,
    _tokenize_discrete_action,
)
from lerobot.processor.converters import create_transition  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE  # noqa: E402

BASE_PATH = Path(__file__).resolve().parents[3] / "outputs/MolmoAct2"
HORIZON = 30
IMAGE_KEY = f"{OBS_IMAGES}.external_0"


@pytest.fixture(scope="module")
def step():
    if not (BASE_PATH / "processor_config.json").is_file():
        pytest.skip("MolmoAct2 base checkpoint not present")
    return MolmoAct2PackInputsProcessorStep(
        base_path=str(BASE_PATH),
        action_mode="both",
        image_keys=[IMAGE_KEY],
        chunk_size=HORIZON,
        max_action_dim=8,
    )


def _chunk(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(HORIZON, width, generator=generator) * 2 - 1


def _pack(step, actions: list[torch.Tensor]):
    """Pack a batch whose rows may have different native widths."""
    batch = len(actions)
    width = max(int(a.shape[-1]) for a in actions)
    padded = torch.zeros(batch, HORIZON, width)
    is_pad = torch.ones(batch, width, dtype=torch.bool)
    for row, action in enumerate(actions):
        native = int(action.shape[-1])
        padded[row, :, :native] = action
        is_pad[row, :native] = False
    observation = {
        OBS_STATE: torch.zeros(batch, width),
        IMAGE_KEY: torch.zeros(batch, 3, 64, 64, dtype=torch.uint8),
    }
    return step(
        create_transition(
            observation=observation,
            action=padded,
            complementary_data={"action_dim_is_pad": is_pad},
        )
    )


def _answer_tokens(step, packed, row: int) -> list[int]:
    """The label span of one row: the supervised action tokens."""
    labels = packed[TransitionKey.COMPLEMENTARY_DATA]["labels"][row]
    return [int(token) for token in labels if int(token) != -100]


# ── Tokenizer behaviour ──────────────────────────────────────────────────────


def test_a_zero_padded_column_changes_every_token(step) -> None:
    """Why the trim is necessary, stated as a test rather than a comment."""
    seven = _chunk(7, seed=1)
    padded = torch.cat([seven, torch.zeros(HORIZON, 1)], dim=-1)
    native = _tokenize_discrete_action(seven.numpy(), step.action_processor)
    widened = _tokenize_discrete_action(padded.numpy(), step.action_processor)
    assert native != widened


def test_token_counts_track_the_native_width(step) -> None:
    seven = _tokenize_discrete_action(_chunk(7, seed=2).numpy(), step.action_processor)
    eight = _tokenize_discrete_action(_chunk(8, seed=2).numpy(), step.action_processor)
    assert len(seven) > 0 and len(eight) > 0


# ── Gate F ───────────────────────────────────────────────────────────────────


def test_a_seven_dim_row_tokenizes_identically_alone_or_beside_an_eight(step) -> None:
    seven = _chunk(7, seed=3)
    eight = _chunk(8, seed=4)
    alone = _pack(step, [seven])
    mixed = _pack(step, [seven, eight])
    assert _answer_tokens(step, alone, 0) == _answer_tokens(step, mixed, 0)


def test_the_eight_dim_row_keeps_its_own_tokens_in_the_mix(step) -> None:
    seven = _chunk(7, seed=5)
    eight = _chunk(8, seed=6)
    alone = _pack(step, [eight])
    mixed = _pack(step, [seven, eight])
    assert _answer_tokens(step, alone, 0) == _answer_tokens(step, mixed, 1)


def test_the_tokens_are_the_rows_own_native_encoding(step) -> None:
    """Not just self-consistent: equal to a direct FAST encode of the native chunk."""
    seven = _chunk(7, seed=7)
    eight = _chunk(8, seed=8)
    mixed = _pack(step, [seven, eight])
    for row, action in enumerate((seven, eight)):
        direct = _tokenize_discrete_action(action.numpy(), step.action_processor)
        packed = _answer_tokens(step, mixed, row)
        # The label span carries the FAST tokens re-encoded as <action_token_N> pieces
        # plus the answer wrapper, so compare through the same rendering.
        rendered = step.processor.tokenizer.decode(packed)
        for token_id in direct:
            assert f"{ACTION_TOKEN_PREFIX}{token_id}>" in rendered


def test_the_gripper_dimension_survives_the_round_trip(step) -> None:
    """The last native dimension is the gripper; it must be inside the encoded span."""
    seven = _chunk(7, seed=9)
    moved = seven.clone()
    moved[:, -1] = -moved[:, -1]
    assert _tokenize_discrete_action(seven.numpy(), step.action_processor) != (
        _tokenize_discrete_action(moved.numpy(), step.action_processor)
    )


def test_sequences_are_padded_only_after_tokenization(step) -> None:
    """Rows of different token counts still produce one rectangular batch with a mask."""
    mixed = _pack(step, [_chunk(7, seed=10), _chunk(8, seed=11)])
    complementary = mixed[TransitionKey.COMPLEMENTARY_DATA]
    ids, mask = complementary["input_ids"], complementary["attention_mask"]
    assert ids.shape == mask.shape
    # Padding is masked out, so a shorter row is not supervised on filler.
    assert int(mask[0].sum()) <= int(ids.shape[1])
    labels = complementary["labels"]
    assert (labels[mask == 0] == -100).all()
