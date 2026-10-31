"""Per-task vocabulary and ladder, from tasks/<name>.json.

``subtasks``: the fixed vocabulary the judge picks from; each entry has an ``id``, ``text``, one-line observable
``criterion``, ``contact`` (annotation/vocab.py slug), ``precision`` (1-5) and ``cursor`` = the index of the policy's
eval_subtasks ``script`` entry it belongs to (-1 = off-task, not mapped). ``stages``: the sim scorer's STAGES
(``name``, cumulative ``value``, ``cursor``) with stage 0 = nothing reached, plus an observable ``criterion``.
A spec with ``objects`` is a template: ``TaskSpec.load("basket", object="tape roll")`` fills ``{object}`` and
``{grasp_contact}`` everywhere."""

import json
from dataclasses import dataclass
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"


@dataclass(frozen=True)
class Subtask:
    id: int
    text: str
    cursor: int
    contact: str
    precision: int
    criterion: str


@dataclass(frozen=True)
class Stage:
    id: int
    name: str
    value: float
    cursor: int
    criterion: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task: str
    script: tuple[str, ...]
    subtasks: tuple[Subtask, ...]
    stages: tuple[Stage, ...]

    @staticmethod
    def names() -> list[str]:
        return sorted(p.stem for p in TASKS_DIR.glob("*.json"))

    @classmethod
    def load(cls, name: str, object: str | None = None) -> "TaskSpec":
        """``name`` = a file stem in tasks/, ``"<stem>:<object>"`` for a template, or ``"sim:<layout>"`` (sim_spec.py)."""
        if name.startswith("sim:"):
            from .sim_spec import spec_for_layout

            return spec_for_layout(name[4:])
        if ":" in name and object is None:
            name, object = name.split(":", 1)
        raw = json.loads((TASKS_DIR / f"{name}.json").read_text())
        if "objects" in raw:
            if object not in raw["objects"]:
                raise ValueError(f"{name}: object must be one of {raw['objects']}, got {object!r}")
            fill = dict(object=object, grasp_contact=raw["grasp_contact"][object])
            raw = json.loads(json.dumps(raw).replace("{object}", fill["object"]).replace("{grasp_contact}", fill["grasp_contact"]))
            raw["name"] = f"{name}:{object}"
        subtasks = tuple(Subtask(**{k: s[k] for k in Subtask.__dataclass_fields__}) for s in raw["subtasks"])
        stages = tuple(Stage(**{k: s[k] for k in Stage.__dataclass_fields__}) for s in raw["stages"])
        spec = cls(name=raw["name"], task=raw["task"], script=tuple(raw["script"]), subtasks=subtasks, stages=stages)
        spec.validate()
        return spec

    def validate(self) -> None:
        assert [s.id for s in self.subtasks] == list(range(len(self.subtasks))), f"{self.name}: subtask ids must be 0..n-1"
        assert [s.id for s in self.stages] == list(range(len(self.stages))), f"{self.name}: stage ids must be 0..n-1"
        assert self.stages[0].value == 0.0, f"{self.name}: stage 0 is 'nothing reached' with value 0"
        values = [s.value for s in self.stages]
        assert values == sorted(values) and values[-1] == 1.0, f"{self.name}: stage values must be increasing and end at 1.0"
        for s in self.subtasks:
            assert -1 <= s.cursor < len(self.script), f"{self.name}: subtask {s.id} cursor {s.cursor} outside the script"
        for s in self.stages:
            assert 0 <= s.cursor < len(self.script), f"{self.name}: stage {s.id} cursor {s.cursor} outside the script"

    def stage_by_name(self, name: str | None) -> Stage:
        """The sim scorer's ``stage`` field (None before the first stage) -> our Stage."""
        if name is None:
            return self.stages[0]
        return next(s for s in self.stages if s.name == name)
