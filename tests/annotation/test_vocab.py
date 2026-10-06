"""Contact and mistake vocabularies (annotation/vocab.py) against their rubrics of record."""

import re
from pathlib import Path

import pytest

from lerobot.annotation.vocab import (
    CONTACT_SLUGS,
    CONTACT_VOCAB,
    MISTAKE_TYPES,
    NA_CODE,
    UNLABELLED,
    code_for,
    phrase_for,
)

RUBRICS = Path(__file__).resolve().parents[2] / "src/lerobot/annotation/rubrics"
RUBRIC = RUBRICS / "contact_strategy_rubric.md"
MISTAKE_RUBRIC = RUBRICS / "quality_mistake_rubric_v2.md"


def _rubric_rows() -> list[tuple[int, str, str, str]]:
    """(code, element, phrase, definition) from the rubric's vocabulary table."""
    rows = []
    for line in RUBRIC.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and re.fullmatch(r"\d+", cells[0]):
            rows.append((int(cells[0]), cells[1], cells[2], cells[3]))
    return rows


def test_fifteen_contiguous_unique_codes():
    assert len(CONTACT_VOCAB) == 15
    assert [e.code for e in CONTACT_VOCAB] == list(range(15))
    assert len({e.slug for e in CONTACT_VOCAB}) == 15
    assert len({e.phrase for e in CONTACT_VOCAB}) == 15
    assert CONTACT_SLUGS == tuple(e.slug for e in CONTACT_VOCAB)


def test_na_and_sentinel():
    assert NA_CODE == 14 and code_for("na") == NA_CODE
    assert phrase_for(NA_CODE) == "not applicable"
    assert UNLABELLED == -1


def test_phrase_and_code_round_trip():
    for element in CONTACT_VOCAB:
        assert code_for(element.slug) == element.code
        assert phrase_for(element.code) == element.phrase


@pytest.mark.parametrize("code", [UNLABELLED, 15])
def test_phrase_for_rejects_codes_outside_the_vocab(code):
    with pytest.raises(ValueError):
        phrase_for(code)


def test_code_for_rejects_unknown_slug():
    with pytest.raises(ValueError):
        code_for("grab")


@pytest.mark.skipif(not RUBRIC.exists(), reason="rubric not present in this checkout")
def test_vocab_matches_the_rubric_table():
    rows = _rubric_rows()
    assert rows == [(e.code, e.slug, e.phrase, e.definition) for e in CONTACT_VOCAB]


@pytest.mark.skipif(not MISTAKE_RUBRIC.exists(), reason="rubric not present in this checkout")
def test_mistake_types_match_the_rubric_table():
    section = MISTAKE_RUBRIC.read_text().split("### 4.1 Types")[1].split("### 4.2")[0]
    assert tuple(re.findall(r"^\| `(\w+)` \|", section, flags=re.M)) == MISTAKE_TYPES
