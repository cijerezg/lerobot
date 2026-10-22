"""Sim eval task registry: scene -> tier, task string (task sheet id), eval_subtasks script, scorer, staged reward.

``LAYOUTS`` is the default set run_sim_eval.sh runs; ``task_for(scene)`` resolves a config ``scene`` (layout
name or path to scenes/<name>/scene.xml). The shell gets the per-scene policy overrides from
``python -c "from lerobot.robots.rebot_b601_sim.tasks import main; main()" overrides <scene>`` (draccus CLI
flags with JSON lists, one per line) and the default layouts from ``... layouts``; config_rl_sim.yaml keeps the nail task as its single values.

Task strings: closest task-sheet entry (lerobot/docs/annotation_notes/rebot_tasks.md). close_lid = T071 verbatim.
bottle_basket and lift_bottle have no sheet entry (sheet_id None): the strings are the shortest natural ones; the
bottle_basket script follows T072 steps 5-8 (grasp / move / release the spray bottle in the basket, return to home)
with T048's "brown basket" wording, lift_bottle's follows T082/T089 (one-step grasp, side-pinch) plus a lift step.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .easy_tasks import BottleBasketScorer, CloseLidScorer, LiftBottleScorer
from .nail_task import NailTaskScorer
from .scorer import TaskScorer

TIERS = ("easy", "hard")


@dataclass(frozen=True)
class Task:
    tier: str
    task: str
    sheet_id: str | None
    eval_subtasks: tuple[str, ...]
    eval_subtask_precisions: tuple[int, ...]
    eval_subtask_contacts: tuple[str, ...]
    scorer: type[TaskScorer]

    @property
    def stages(self) -> list[tuple[str, float, int]]:
        return list(self.scorer.STAGES)

    def overrides(self) -> list[str]:
        """draccus CLI overrides of the policy fields config_rl_sim.yaml carries as the nail task."""
        return [
            f"--policy.task={self.task}",
            f"--policy.eval_subtasks={json.dumps(list(self.eval_subtasks))}",
            f"--policy.eval_subtask_precisions={json.dumps(list(self.eval_subtask_precisions))}",
            f"--policy.eval_subtask_contacts={json.dumps(list(self.eval_subtask_contacts))}",
        ]


NAIL = dict(
    task="Put the nail clippers in the clipper case", sheet_id="T085",
    eval_subtasks=("grasp the large nail clipper", "move the large nail clipper to the clipper case",
                   "release the large nail clipper in the large slot", "push the large nail clipper into the large slot",
                   "grasp the nail clipper", "move the nail clipper to the clipper case",
                   "release the nail clipper in the small slot", "push the nail clipper into the small slot", "return to home"),
    eval_subtask_precisions=(4, 4, 5, 3, 4, 4, 5, 3, 1),
    eval_subtask_contacts=("top-pinch", "na", "insert", "push", "top-pinch", "na", "insert", "push", "na"),
    scorer=NailTaskScorer,
)

TASKS: dict[str, Task] = {
    "clippers_only": Task(tier="easy", **NAIL),
    "ep0": Task(tier="hard", **NAIL),
    "ep1": Task(tier="hard", **NAIL),
    "ep2": Task(tier="hard", **NAIL),
    "ep3": Task(tier="hard", **NAIL),
    "bottle_basket": Task(
        tier="easy", task="Put the bottle in the brown basket", sheet_id=None,
        eval_subtasks=("grasp the bottle", "move the bottle to the brown basket", "release the bottle in the brown basket", "return to home"),
        eval_subtask_precisions=(3, 1, 1, 1), eval_subtask_contacts=("side-pinch", "na", "set-down", "na"),
        scorer=BottleBasketScorer,
    ),
    "lift_bottle": Task(
        tier="easy", task="Lift the bottle", sheet_id=None,
        eval_subtasks=("grasp the bottle", "lift the bottle"),
        eval_subtask_precisions=(3, 1), eval_subtask_contacts=("side-pinch", "na"),
        scorer=LiftBottleScorer,
    ),
    "close_lid": Task(
        tier="easy", task="Close the headphone case lid", sheet_id="T071",
        eval_subtasks=("grasp the headphone case lid", "close the headphone case lid", "return to home"),
        eval_subtask_precisions=(3, 2, 1), eval_subtask_contacts=("rim-pinch", "rim-pinch", "na"),
        scorer=CloseLidScorer,
    ),
}

# The default eval set (run_sim_eval.sh LAYOUTS): easy tier first, then the hard tier. ep1/ep2 stay on disk.
LAYOUTS = ("clippers_only", "bottle_basket", "close_lid", "lift_bottle", "ep0", "ep3")


def layout_name(scene: str) -> str:
    """The registry key of a config ``scene``: a layout name, or scenes/<name>/scene.xml."""
    return Path(scene).parent.name if scene.endswith(".xml") else scene


def task_for(scene: str) -> Task:
    return TASKS[layout_name(scene)]


def main() -> None:
    """``layouts`` -> the default set; ``overrides <scene>`` -> policy CLI flags, one per line (run_sim_eval.sh)."""
    if sys.argv[1] == "layouts":
        print(" ".join(LAYOUTS))
    elif sys.argv[1] == "overrides":
        print("\n".join(task_for(sys.argv[2]).overrides()))
