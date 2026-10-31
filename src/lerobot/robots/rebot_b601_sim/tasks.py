"""Sim eval task registry: scene -> tier, task string (task sheet id), eval_subtasks script, scorer, staged reward.

``LAYOUTS`` is the default set run_sim_eval.sh runs; ``task_for(scene)`` resolves a config ``scene`` (layout
name or path to scenes/<name>/scene.xml). The shell gets the per-scene policy overrides from
``python -c "from lerobot.robots.rebot_b601_sim.tasks import main; main()" overrides <scene>`` (draccus CLI
flags with JSON lists, one per line) and the default layouts from ``... layouts``; config_rl_sim.yaml keeps the nail task as its single values.

Placement variants: scenes/<layout>_v<k>/ (scenes/README.md, migration/sim_variants_2026-10-28) share the base layout's Task;
``layout_name`` strips the suffix (``_v<k>``, or ``_p<k>`` for the twin placements), ``variants(layout)`` lists them on disk and
``eval_layouts()`` is LAYOUTS with every base that has variants replaced by them (``layouts`` prints that, so run_sim_eval.sh
runs the variants instead of the base).

Task strings: closest task-sheet entry (lerobot/docs/annotation_notes/rebot_tasks.md). close_lid = T071 verbatim.
bottle_basket and lift_bottle have no sheet entry (sheet_id None): the strings are the shortest natural ones; the
bottle_basket script follows T072 steps 5-8 (grasp / move / release the spray bottle in the basket, return to home)
with T048's "brown basket" wording, lift_bottle's follows T082/T089 (one-step grasp, side-pinch) plus a lift step.

Twin set (2026-10-30, scenes twin_l<L>_<config>_p<1-4>, migration/digital_twin_2026-10-29/envs/ENVS_REPORT.md): four levels
twin_l1 .. twin_l4 on the digital-twin desk (single object / target + distractors / full desk / full desk sequential), four
hand-designed configurations per level, four placements each (p1 = the configuration, p2-p4 jittered), dense staged scorers
(twin_tasks.py). ``TWIN=1`` in the environment (or ``layouts --twin``) makes ``eval_layouts()`` return the 64 twin placements
instead of the default set; run_sim_eval.sh passes it through. Task strings: T054 / T038 verbatim, pill bottle after T051 steps
1-3, orange bottle = T107's name for the ROAR bottle, the sequence tasks after T037 steps 1-6 (first object first), the fold
task = T067 steps 1-5 ("Close the bit kit"), the lid task = close_lid's T071 wiring on the twin's headphone case (no scene yet).
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .easy_tasks import BottleBasketScorer, CloseLidScorer, LiftBottleScorer
from .nail_task import NailTaskScorer
from .scorer import TaskScorer
from .twin_tasks import FoldKitScorer, pick_place_scorer, sequence_scorer

TIERS = ("easy", "hard", "twin_l1", "twin_l2", "twin_l3", "twin_l4")
TWIN_TIERS = TIERS[2:]
TIER_TIME_S = {"twin_l1": 30.0, "twin_l2": 45.0, "twin_l3": 90.0, "twin_l4": 120.0}  # episode budget per twin tier (user 2026-10-30);
# res600 run at 60 s: L1 successes finished by 25 s, L2 by 50 s (one), L4 sequences cannot finish in 60 s. Old tiers keep the config's value.
SCENES_DIR = Path(__file__).resolve().parent / "scenes"


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
        ] + ([f"--env.processor.reset.control_time_s={TIER_TIME_S[self.tier]}"] if self.tier in TIER_TIME_S else [])


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


# Twin basket tasks: scene body -> (noun in the task strings, sheet id, grasp contact, "return to home" step)
_TWIN_BASKET = {
    "tape_roll": ("the tape roll", "T054", "rim-pinch", False),
    "tape_roll_2": ("the other tape roll", None, "rim-pinch", False),
    "paper_cup": ("the cup", "T038", "rim-pinch", True),
    "pill_bottle": ("the pill bottle", None, "side-pinch", True),
    "roar_bottle": ("the orange bottle", None, "side-pinch", True),
}


def _steps(body: str) -> tuple[list[str], str]:
    noun, _, contact, _ = _TWIN_BASKET[body]
    return [f"grasp {noun}", f"move {noun} to the basket", f"release {noun} in the basket"], contact


def _basket_task(tier: str, body: str) -> Task:
    noun, sheet_id, contact, home = _TWIN_BASKET[body]
    steps, _ = _steps(body)
    steps += ["return to home"] if home else []
    cursors = (0, 0, 1, 1, 2, 3 if home else 2)  # approach, contact, lifted, transport, over, inside
    return Task(tier=tier, task=f"Put {noun} in the basket", sheet_id=sheet_id, eval_subtasks=tuple(steps),
                eval_subtask_precisions=(3, 1, 1, 1)[: len(steps)], eval_subtask_contacts=(contact, "na", "drop", "na")[: len(steps)],
                scorer=pick_place_scorer(body, cursors))


def _sequence_task(tier: str, task: str, bodies: tuple[str, ...]) -> Task:
    """"Put the <a> and the <b> in the basket": the single-object steps in order plus "return to home"; object i's stages point at
    steps 3i (approach, contact), 3i + 1 (lifted, transport), 3i + 2 (over) and 3i + 3 (inside = the next grasp, or home)."""
    steps, precisions, contacts = [], [], []
    for body in bodies:
        st, contact = _steps(body)
        steps += st; precisions += [3, 1, 1]; contacts += [contact, "na", "drop"]
    steps += ["return to home"]; precisions += [1]; contacts += ["na"]
    cursors = tuple((3 * i, 3 * i, 3 * i + 1, 3 * i + 1, 3 * i + 2, 3 * i + 3) for i in range(len(bodies)))
    return Task(tier=tier, task=task, sheet_id=None, eval_subtasks=tuple(steps), eval_subtask_precisions=tuple(precisions),
                eval_subtask_contacts=tuple(contacts), scorer=sequence_scorer(bodies, cursors))


TASKS.update({
    "twin_l1_tape_basket_right": _basket_task("twin_l1", "tape_roll"),
    "twin_l1_tape_basket_left": _basket_task("twin_l1", "tape_roll"),
    "twin_l1_pill_basket_far": _basket_task("twin_l1", "pill_bottle"),
    "twin_l1_roar_basket_near": _basket_task("twin_l1", "roar_bottle"),
    "twin_l2_tape_bits": _basket_task("twin_l2", "tape_roll"),
    "twin_l2_pill_case": _basket_task("twin_l2", "pill_bottle"),
    "twin_l2_roar_tapes": _basket_task("twin_l2", "roar_bottle"),
    "twin_l2_tape_bottles": _basket_task("twin_l2", "tape_roll"),
    "twin_l3_tape": _basket_task("twin_l3", "tape_roll"),
    "twin_l3_pill": _basket_task("twin_l3", "pill_bottle"),
    "twin_l3_fold_kit": Task(
        tier="twin_l3", task="Close the bit kit", sheet_id=None,
        eval_subtasks=("grasp the outer edge of the flap with the bits", "fold the flap with the bits over the middle of the kit",
                       "grasp the outer edge of the flap with the white card", "fold the flap with the white card over the folded flap",
                       "press the kit flat", "return to home"),
        eval_subtask_precisions=(3, 2, 3, 2, 1, 1), eval_subtask_contacts=("cloth-pinch", "drop", "cloth-pinch", "set-down", "press", "na"),
        scorer=FoldKitScorer,
    ),
    # the twin's headphone_case / headphone_lid / lid_hinge are the close_lid asset: same scorer and script (no scene folder yet,
    # see ENVS_REPORT.md open points; TWIN_LAYOUTS lists only layouts with placements on disk)
    "twin_l3_close_lid": Task(tier="twin_l3", **{k: getattr(TASKS["close_lid"], k) for k in ("task", "sheet_id", "eval_subtasks", "eval_subtask_precisions", "eval_subtask_contacts", "scorer")}),
    "twin_l4_tape_cup": _sequence_task("twin_l4", "Put the tape roll and the cup in the basket", ("tape_roll", "paper_cup")),
    "twin_l4_two_tapes": _sequence_task("twin_l4", "Put the two tape rolls in the basket", ("tape_roll", "tape_roll_2")),
    "twin_l4_pill_roar": _sequence_task("twin_l4", "Put the pill bottle and the orange bottle in the basket", ("pill_bottle", "roar_bottle")),
    "twin_l4_three": _sequence_task("twin_l4", "Put the tape roll, the pill bottle and the cup in the basket", ("tape_roll", "pill_bottle", "paper_cup")),
})

# The default eval set (run_sim_eval.sh LAYOUTS): easy tier first, then the hard tier. ep1/ep2 stay on disk.
LAYOUTS = ("clippers_only", "bottle_basket", "close_lid", "ep0", "ep3")
# The twin set (TWIN=1): levels in order; a twin layout runs through its placements twin_<...>_p<k> on disk (``variants``).
TWIN_LAYOUTS = tuple(name for name in TASKS if name.startswith("twin_"))


def layout_name(scene: str) -> str:
    """The registry key of a config ``scene``: a layout name, or scenes/<name>/scene.xml; a placement variant
    ``<layout>_v<k>`` resolves to its base layout."""
    name = Path(scene).parent.name if scene.endswith(".xml") else scene
    return re.sub(r"_[vp]\d+$", "", name)


def variants(layout: str) -> list[str]:
    """The placement variants ``<layout>_v<k>`` (old set) or placements ``<layout>_p<k>`` (twin set) on disk under SCENES_DIR, by k."""
    names = [p.name for p in SCENES_DIR.glob(f"{layout}_[vp]*") if re.fullmatch(rf"{re.escape(layout)}_[vp]\d+", p.name) and (p / "scene.xml").exists()]
    return sorted(names, key=lambda n: int(n[len(layout) + 2:]))


def eval_layouts(twin: bool | None = None) -> list[str]:
    """LAYOUTS (or TWIN_LAYOUTS when ``twin``, default: the TWIN env var) with each base layout replaced by its variants when it
    has any (the base itself is then not run); a twin layout without placements on disk is skipped."""
    if twin is None:
        twin = os.environ.get("TWIN", "0") not in ("", "0")
    if twin:
        return [name for layout in TWIN_LAYOUTS for name in variants(layout)]
    return [name for layout in LAYOUTS for name in (variants(layout) or [layout])]


def task_for(scene: str) -> Task:
    return TASKS[layout_name(scene)]


def main() -> None:
    """``layouts [--twin]`` -> the default (or twin) set (variants in place of their base); ``overrides <scene>`` -> policy CLI flags,
    one per line (run_sim_eval.sh)."""
    if sys.argv[1] == "layouts":
        print(" ".join(eval_layouts(True if "--twin" in sys.argv[2:] else None)))
    elif sys.argv[1] == "overrides":
        print("\n".join(task_for(sys.argv[2]).overrides()))


if __name__ == "__main__":
    main()
