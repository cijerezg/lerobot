"""Turns the model's text into a validated Judgement or raises JudgeParseError."""

import json
import re
from dataclasses import asdict, dataclass

from .prompt import ANSWER_KEYS
from .task_spec import TaskSpec


class JudgeParseError(ValueError):
    pass


@dataclass
class Judgement:
    subtask_id: int
    stage_id: int
    progress: float
    confidence: float
    evidence: str
    subtask: str = ""
    stage: str = ""
    cursor: int = 0  # script index of the subtask (-1 = off-task)
    stage_value: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m is None:
        raise JudgeParseError(f"no JSON object in: {text[:200]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise JudgeParseError(f"bad JSON: {e}: {text[:200]!r}") from e


def _int(raw: dict, key: str, n: int) -> int:
    v = raw[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v):
        raise JudgeParseError(f"{key} must be an integer, got {v!r}")
    if not 0 <= int(v) < n:
        raise JudgeParseError(f"{key} {int(v)} outside 0..{n - 1}")
    return int(v)


def _unit(raw: dict, key: str) -> float:
    v = raw[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
        raise JudgeParseError(f"{key} must be a number in [0, 1], got {v!r}")
    return float(v)


def parse_response(text: str, spec: TaskSpec) -> Judgement:
    raw = _extract_json(text)
    if not isinstance(raw, dict):
        raise JudgeParseError(f"expected an object, got {type(raw).__name__}")
    missing = [k for k in ANSWER_KEYS if k not in raw]
    if missing:
        raise JudgeParseError(f"missing keys {missing}")
    if not isinstance(raw["evidence"], str):
        raise JudgeParseError("evidence must be a string")
    sub = spec.subtasks[_int(raw, "subtask_id", len(spec.subtasks))]
    stage = spec.stages[_int(raw, "stage_id", len(spec.stages))]
    return Judgement(
        subtask_id=sub.id, stage_id=stage.id, progress=_unit(raw, "progress_0_1"), confidence=_unit(raw, "confidence"),
        evidence=raw["evidence"].strip(), subtask=sub.text, stage=stage.name, cursor=sub.cursor, stage_value=stage.value,
    )
