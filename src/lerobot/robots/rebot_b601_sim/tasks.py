"""Sim eval task registry: scene -> tier, task string (task sheet id), eval_subtasks script, scorer, staged reward.

``LAYOUTS`` is the eval set run_sim_eval.sh runs: the twin set (below). ``task_for(scene)`` resolves a config ``scene``
(layout name or path to scenes/<name>/scene.xml). The shell gets the per-scene policy overrides from
``python -c "from lerobot.robots.rebot_b601_sim.tasks import main; main()" overrides <scene>`` (draccus CLI
flags with JSON lists, one per line) and the layouts from ``... layouts``; config_rl_sim.yaml keeps the nail task as its single values.

Placements: scenes/<layout>_p<k>/ (twin set) or scenes/<layout>_v<k>/ (the old photo-composite set, scripts/sim_builders/variants)
share the base layout's Task; ``layout_name`` strips the suffix, ``variants(layout)`` lists them on disk and ``eval_layouts()`` is
LAYOUTS with every base replaced by its placements (``layouts`` prints that, so run_sim_eval.sh runs the placements, never a base).

The old set (clippers_only, bottle_basket, close_lid, ep0, ep3 and their ``_v<k>`` variants: photo table texture composited onto
a real background) stays registered so earlier results (sim_summary, compare_probes) still resolve, but is no longer run
(2026-11-05: the twin desk is the eval set).

Task strings: closest task-sheet entry (lerobot/docs/annotation_notes/rebot_tasks.md). close_lid = T071 verbatim.
bottle_basket and lift_bottle have no sheet entry (sheet_id None): the strings are the shortest natural ones; the
bottle_basket script follows T072 steps 5-8 (grasp / move / release the spray bottle in the basket, return to home)
with T048's "brown basket" wording, lift_bottle's follows T082/T089 (one-step grasp, side-pinch) plus a lift step.

Twin set (2026-10-30, scenes twin_l<L>_<config>_p<1-4>, scripts/sim_builders/digital_twin/envs/ENVS_REPORT.md): four levels
twin_l1 .. twin_l4 on the digital-twin desk (single object / target + distractors / full desk / full desk sequential), four
hand-designed configurations per level, four hand-designed placements each (v2 2026-11-08, scripts/sim_builders/twin_eval_v2; p2-p4 were +-6 cm jitters of p1 until then), dense staged scorers
(twin_tasks.py). ``eval_layouts()`` returns its placements on disk (a configuration without a scene folder is skipped). Task strings: T054 / T038 verbatim, pill bottle after T051 steps
1-3, orange bottle = T107's name for the ROAR bottle, the bit after T056/T057 ("the bit", precision 5 top-pinch), the sequence
tasks after T037 steps 1-6 (first object first; "the two bits" in any order), the fold task = T067 steps 1-5 ("Close the bit
kit", also alone on the desk at L1 with the L3 time budget), the lid task = close_lid's T071 wiring on the twin's headphone case.

Tube set (2026-11-15, scenes tube_peg_{left,right,two,three}_p1, scripts/sim_builders/tube_peg/README.md): one, two or three tubes, each
with its own peg holder, on the twin desk for the v2 tube policy, tier ``tube``, scorer tube_task.py. Not in LAYOUTS (run with LAYOUTS="$(... tube_layouts)"). Task and
subtask strings are the tube recordings' ("put the tube in the peg", "... on the left / right", "return home"); the layouts also
set two robot fields (``Task.robot``): the finger force cap and the joint zero correction. Start levels (2026-11-25, scenes
tube_peg_{left,right}_{half,near}_p1, tiers ``tube_half`` / ``tube_near``): the same scenes with the arm starting halfway to the tube or
just above it; part of ``tube_layouts`` (8 scenes), the three levels alone are LAYOUTS="$(... tube_start_layouts)".

Choice set (2026-11-17, scenes choice_<pair>_ask_<object>_p<1-6>, scripts/sim_builders/object_choice): the basket and two objects, the
task names one ("Put the banana in the basket"), tier ``choice``, scorer choice_tasks.py (the named object's ladder + which object
the arm went to). Not in LAYOUTS (run with LAYOUTS="$(... choice_layouts)"). The banana and the mug are not in the training data;
both instructions of a pair use the same contact word, so they differ in the noun only.

Small objects set (2026-11-26, scenes {die_cup,plug_box,screwdriver_case,nut_peg}_<config>_p1, scripts/sim_builders/small_objects):
the four families of the small-objects recordings on the twin desk, four configurations each (three with one object, one with
several), tier ``small``. Start levels (scenes <layout>_{near,turned}_p1, tiers ``small_near`` / ``small_turned``,
build_small_starts.py): the same scenes with the arm starting near the first object (on top of it, close or midway to its left or right) and moving toward it, and so with the wrist turned 45 deg;
``small_layouts`` = all 48 scenes. Not in LAYOUTS (run with LAYOUTS="$(... small_layouts)"). Task string as recorded ("tidy up the table"),
subtask strings in the wording of annotation/rubrics/annotation_rubric.md ("approach die", "grasp die", "move die to cup",
"release die", ..., "return home"). Die, plug and screwdriver = the pick-and-place ladder with the place given by a marker box of
the container (the cup's inside, the plug's colour compartment, the case tray); the nut = the tube ladder on the holder's pin.
Robot fields as the tube set.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .choice_tasks import choice_scorer
from .easy_tasks import BottleBasketScorer, CloseLidScorer, LiftBottleScorer
from .nail_task import NailTaskScorer
from .scorer import TaskScorer
from .tube_task import tube_peg_scorer
from .twin_tasks import FoldKitScorer, pick_place_scorer, sequence_scorer

TIERS = ("easy", "hard", "twin_l1", "twin_l2", "twin_l3", "twin_l4", "tube", "choice", "tube_half", "tube_near", "small", "small_near", "small_turned")
TWIN_TIERS = TIERS[2:6]
TIER_TIME_S = {"twin_l1": 30.0, "twin_l2": 45.0, "twin_l3": 90.0, "twin_l4": 120.0, "tube": 45.0, "choice": 30.0, "small": 30.0}  # episode budget per twin tier (user 2026-10-30);
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
    time_s: float | None = None  # episode budget; None = the tier's (TIER_TIME_S)
    robot: tuple[tuple[str, object], ...] = ()  # env.robot fields this layout sets (name, value)

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
        ] + ([f"--env.processor.reset.control_time_s={self.time_s or TIER_TIME_S[self.tier]}"] if self.time_s or self.tier in TIER_TIME_S else []) + [
            f"--env.robot.{name}={json.dumps(value)}" for name, value in self.robot]


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
    # hex bits: T056/T057 wording, "grasp the bit" annotated precision 5 / top-pinch there; no return step (as T056)
    "bit_1": ("the bit", None, "top-pinch", False),
    "bit_2": ("the other bit", None, "top-pinch", False),
    # choice set only (not in the training data); side-pinch as the bottles they are paired with
    "banana": ("the banana", None, "side-pinch", True),
    "mug": ("the mug", None, "side-pinch", True),
}
_GRASP_PRECISION = {"bit_1": 5, "bit_2": 5}  # grasp step precision; 3 otherwise


def _steps(body: str) -> tuple[list[str], str]:
    noun, _, contact, _ = _TWIN_BASKET[body]
    return [f"grasp {noun}", f"move {noun} to the basket", f"release {noun} in the basket"], contact


def _basket_task(tier: str, body: str) -> Task:
    noun, sheet_id, contact, home = _TWIN_BASKET[body]
    steps, _ = _steps(body)
    steps += ["return to home"] if home else []
    cursors = (0, 0, 1, 1, 2, 3 if home else 2)  # approach, contact, lifted, transport, over, inside
    return Task(tier=tier, task=f"Put {noun} in the basket", sheet_id=sheet_id, eval_subtasks=tuple(steps),
                eval_subtask_precisions=(_GRASP_PRECISION.get(body, 3), 1, 1, 1)[: len(steps)],
                eval_subtask_contacts=(contact, "na", "drop", "na")[: len(steps)], scorer=pick_place_scorer(body, cursors))


def _sequence_task(tier: str, task: str, bodies: tuple[str, ...], any_order: bool = False) -> Task:
    """"Put the <a> and the <b> in the basket": the single-object steps in order plus "return to home"; object i's stages point at
    steps 3i (approach, contact), 3i + 1 (lifted, transport), 3i + 2 (over) and 3i + 3 (inside = the next grasp, or home).
    ``any_order``: identical objects (the two bits), the scorer takes them in whatever order the policy picks them."""
    steps, precisions, contacts = [], [], []
    for body in bodies:
        st, contact = _steps(body)
        steps += st; precisions += [_GRASP_PRECISION.get(body, 3), 1, 1]; contacts += [contact, "na", "drop"]
    steps += ["return to home"]; precisions += [1]; contacts += ["na"]
    cursors = tuple((3 * i, 3 * i, 3 * i + 1, 3 * i + 1, 3 * i + 2, 3 * i + 3) for i in range(len(bodies)))
    return Task(tier=tier, task=task, sheet_id=None, eval_subtasks=tuple(steps), eval_subtask_precisions=tuple(precisions),
                eval_subtask_contacts=tuple(contacts), scorer=sequence_scorer(bodies, cursors, any_order=any_order))


FOLD_KIT = dict(
    task="Close the bit kit", sheet_id=None,
    eval_subtasks=("grasp the outer edge of the flap with the bits", "fold the flap with the bits over the middle of the kit",
                   "grasp the outer edge of the flap with the white card", "fold the flap with the white card over the folded flap",
                   "press the kit flat", "return to home"),
    eval_subtask_precisions=(3, 2, 3, 2, 1, 1), eval_subtask_contacts=("cloth-pinch", "drop", "cloth-pinch", "set-down", "press", "na"),
    scorer=FoldKitScorer,
)


TASKS.update({
    # 2026-11-08 bits pass (user: the set was tape heavy): tape_basket_left, roar_basket_near, tape_bits and two_tapes replaced by
    # bit_single, kit_alone, bit_clutter and two_bits; three reordered so the tape is last
    "twin_l1_tape_basket_right": _basket_task("twin_l1", "tape_roll"),
    "twin_l1_bit_single": _basket_task("twin_l1", "bit_1"),
    "twin_l1_pill_basket_far": _basket_task("twin_l1", "pill_bottle"),
    "twin_l1_kit_alone": Task(tier="twin_l1", time_s=TIER_TIME_S["twin_l3"], **FOLD_KIT),  # the fold needs the L3 budget
    "twin_l2_bit_clutter": _basket_task("twin_l2", "bit_1"),
    "twin_l2_pill_case": _basket_task("twin_l2", "pill_bottle"),
    "twin_l2_roar_tapes": _basket_task("twin_l2", "roar_bottle"),
    "twin_l2_tape_bottles": _basket_task("twin_l2", "tape_roll"),
    "twin_l3_tape": _basket_task("twin_l3", "tape_roll"),
    "twin_l3_pill": _basket_task("twin_l3", "pill_bottle"),
    "twin_l3_fold_kit": Task(tier="twin_l3", **FOLD_KIT),
    # the twin's headphone_case / headphone_lid / lid_hinge are the close_lid asset: same scorer and script
    "twin_l3_close_lid": Task(tier="twin_l3", **{k: getattr(TASKS["close_lid"], k) for k in ("task", "sheet_id", "eval_subtasks", "eval_subtask_precisions", "eval_subtask_contacts", "scorer")}),
    "twin_l4_tape_cup": _sequence_task("twin_l4", "Put the tape roll and the cup in the basket", ("tape_roll", "paper_cup")),
    "twin_l4_two_bits": _sequence_task("twin_l4", "Put the two bits in the basket", ("bit_1", "bit_2"), any_order=True),
    "twin_l4_pill_roar": _sequence_task("twin_l4", "Put the pill bottle and the orange bottle in the basket", ("pill_bottle", "roar_bottle")),
    "twin_l4_three": _sequence_task("twin_l4", "Put the pill bottle, the cup and the tape roll in the basket", ("pill_bottle", "paper_cup", "tape_roll")),
})

# Tube set. Left / right as in the recordings' labels = the robot's own (left = +y = right in the top image: 14 of 16 "left" grasps
# of the tube roots are at y > 0). The sides in script order as in the recordings: left, right, then the dark peg. Robot fields: the rigid tube stays in the pads only up to ~10 N per finger, and the real
# grasp states put the uncorrected sim fingers 22-52 mm above the table (shoulder_lift zero, scripts/sim_builders/tube_peg/README.md).
_TUBE_ROBOT = (("gripper_force_cap", 10.0), ("joint_zero_offset_deg", (0.0, -3.43, 0.0, 0.0, 0.0, 0.0, 0.0)))


_TUBE_STEPS = {
    "left": ("approach tube on the left", "grasp tube on the left", "move left tube to peg on the left", "insert tube in peg on the left"),
    "right": ("approach tube on the right", "grasp tube on the right", "move right tube to peg on the right", "insert tube in peg on the right"),
    "dark": ("approach tube at the dark peg", "grasp tube at the dark peg", "move tube to the dark peg", "insert tube in the dark peg"),
}


def _tube_task(*sides: str, tier: str = "tube") -> Task:
    """Tube i of the scene (``tube``, ``tube_2``, ...) is the one ``sides[i]`` names; 45 s per tube (the recordings take ~20 s each)."""
    n = len(sides)
    steps = tuple(step for side in sides for step in _TUBE_STEPS[side]) + ("return home",)
    return Task(tier=tier, task="put the tube in the peg", sheet_id=None, eval_subtasks=steps, eval_subtask_precisions=(3, 3, 3, 3) * n + (1,),
                eval_subtask_contacts=("na", "top-pinch", "na", "insert") * n + ("na",), scorer=tube_peg_scorer(n), time_s=45.0 * n, robot=_TUBE_ROBOT)


TASKS.update({"tube_peg_left": _tube_task("left"), "tube_peg_right": _tube_task("right"),
              "tube_peg_two": _tube_task("left", "right"), "tube_peg_three": _tube_task("left", "right", "dark")})
# Start levels (2026-11-25, scripts/sim_builders/tube_peg/build_tube_starts.py): the single-tube layouts with the arm starting
# halfway to the tube (tier ``tube_half``) or at the end of the approach, just above it (``tube_near``), instead of at the park
# pose. Same scene, task and scorer; the scene file carries the start pose (``start_deg``). TUBE_START_LAYOUTS = the three levels.
TASKS.update({f"tube_peg_{side}_{level}": _tube_task(side, tier=f"tube_{level}") for side in ("left", "right") for level in ("half", "near")})
TUBE_START_LAYOUTS = tuple(f"tube_peg_{side}{level}" for side in ("left", "right") for level in ("", "_half", "_near"))
# The tube eval set of every checkpoint: the four layouts from the park pose + the single-tube layouts at the two closer starts.
TUBE_LAYOUTS = ("tube_peg_left", "tube_peg_right", "tube_peg_two", "tube_peg_three") + tuple(name for name in TUBE_START_LAYOUTS if name.endswith(("_half", "_near")))

# Small objects set: family -> (object noun, the move step's place, the place's marker box in the scene, None = the pin of nut_holder).
# Objects of a scene: ``die``, ``die_2``, ... / ``screwdriver``, ... / ``nut``, ... / ``plug_<colour>``.
_SMALL = {
    "die": ("die", "to cup", "dice_cup_region"),
    "screwdriver": ("screwdriver", "to case", "screwdriver_case_region"),
    "nut": ("nut", "to peg", None),
    **{f"plug_{c}": (f"{c} plug", f"with the other {c} plugs", f"plug_box_region_{c}") for c in ("yellow", "red", "blue")},
}


def _small_task(*objects: str, tier: str = "small") -> Task:
    """``objects`` = the family of each object in script order (one family, or plugs of several colours). 30 s per object, 45 s per
    nut. Identical objects are taken in any order; the nuts stack on the one pin."""
    n = len(objects)
    steps = tuple(step for o in objects for noun, place, _ in [_SMALL[o]]
                  for step in (f"approach {noun}", f"grasp {noun}", f"move {noun} {place}", f"release {noun}")) + ("return home",)
    bodies = tuple(o if o.startswith("plug") or i == 0 else f"{o}_{i + 1}" for i, o in enumerate(objects))
    cursors = tuple(tuple(4 * i + c for c in (1, 1, 2, 2, 3, 4)) for i in range(n))  # approach, contact, lifted, transport, over, inside
    regions = tuple(_SMALL[o][2] for o in objects)
    if objects[0] == "nut":
        scorer = tube_peg_scorer(n, tube="nut", holder="nut_holder", bore_r=0.0022, one_holder=True)  # bore = small_objects.NUT_HOLE_R
    elif n == 1:
        scorer = pick_place_scorer(bodies[0], cursors[0], region=regions[0])
    else:
        scorer = sequence_scorer(bodies, cursors, any_order=len(set(objects)) == 1, regions=regions)
    return Task(tier=tier, task="tidy up the table", sheet_id=None, eval_subtasks=steps, eval_subtask_precisions=(3, 3, 3, 3) * n + (1,),
                eval_subtask_contacts=("na", "top-pinch", "na", "insert" if objects[0] == "nut" else "drop") * n + ("na",), scorer=scorer,
                time_s=(45.0 if objects[0] == "nut" else 30.0) * n, robot=_TUBE_ROBOT)


_SMALL_LAYOUTS = {
    "die_cup_left": ("die",), "die_cup_right": ("die",), "die_cup_across": ("die",), "die_cup_three": ("die", "die", "die"),
    "plug_box_yellow": ("plug_yellow",), "plug_box_blue": ("plug_blue",), "plug_box_red": ("plug_red",), "plug_box_three": ("plug_yellow", "plug_red", "plug_blue"),
    "screwdriver_case_left": ("screwdriver",), "screwdriver_case_right": ("screwdriver",), "screwdriver_case_across": ("screwdriver",), "screwdriver_case_two": ("screwdriver", "screwdriver"),
    "nut_peg_left": ("nut",), "nut_peg_right": ("nut",), "nut_peg_across": ("nut",), "nut_peg_two": ("nut", "nut"),
}
# Start levels (2026-11-26, scripts/sim_builders/small_objects/build_small_starts.py): every layout also with the arm starting near
# its first object (on top of it, or close / midway to one side) and moving toward it (tier ``small_near``), and so with the wrist turned 45 deg (``small_turned``); the scene file carries the
# start pose (``start_deg``). SMALL_LAYOUTS = the 16 layouts x 3 starts, a layout's three starts together.
TASKS.update({name + level: _small_task(*objects, tier="small" + level) for name, objects in _SMALL_LAYOUTS.items() for level in ("", "_near", "_turned")})
SMALL_LAYOUTS = tuple(name for name, task in TASKS.items() if task.tier.startswith("small"))


def _choice_task(asked: str, other: str) -> Task:
    task = _basket_task("choice", asked)
    return Task(**{k: getattr(task, k) for k in ("tier", "task", "sheet_id", "eval_subtasks", "eval_subtask_precisions", "eval_subtask_contacts")},
                scorer=choice_scorer(asked, other, (0, 0, 1, 1, 2, 3)))


_CHOICE_PAIRS = {"pill_roar": ("pill_bottle", "roar_bottle"), "pill_banana": ("pill_bottle", "banana"), "pill_mug": ("pill_bottle", "mug")}
TASKS.update({f"choice_{pair}_ask_{asked.split('_')[0]}": _choice_task(asked, other)
              for pair, (a, b) in _CHOICE_PAIRS.items() for asked, other in ((a, b), (b, a))})
CHOICE_LAYOUTS = tuple(name for name in TASKS if name.startswith("choice_"))

# The eval set (run_sim_eval.sh): the twin levels in order; a layout runs through its placements twin_<...>_p<k> on disk (``variants``).
LAYOUTS = tuple(name for name in TASKS if name.startswith("twin_"))


def layout_name(scene: str) -> str:
    """The registry key of a config ``scene``: a layout name, or scenes/<name>/scene.xml; a placement variant
    ``<layout>_v<k>`` resolves to its base layout."""
    name = Path(scene).parent.name if scene.endswith(".xml") else scene
    return re.sub(r"_[vp]\d+$", "", name)


def variants(layout: str) -> list[str]:
    """The placement variants ``<layout>_v<k>`` (old set) or placements ``<layout>_p<k>`` (twin set) on disk under SCENES_DIR, by k."""
    names = [p.name for p in SCENES_DIR.glob(f"{layout}_[vp]*") if re.fullmatch(rf"{re.escape(layout)}_[vp]\d+", p.name) and (p / "scene.xml").exists()]
    return sorted(names, key=lambda n: int(n[len(layout) + 2:]))


def eval_layouts(layouts: tuple[str, ...] | None = None) -> list[str]:
    """``layouts`` (default LAYOUTS) with each layout replaced by its placements on disk; a layout without any is skipped (never run as a base)."""
    return [name for layout in (LAYOUTS if layouts is None else layouts) for name in variants(layout)]


def task_for(scene: str) -> Task:
    return TASKS[layout_name(scene)]


def main() -> None:
    """``layouts`` -> the eval set (placements in place of their layout); ``overrides <scene>`` -> policy CLI flags,
    one per line (run_sim_eval.sh)."""
    if sys.argv[1] == "layouts":
        print(" ".join(eval_layouts()))
    elif sys.argv[1] == "tube_layouts":
        print(" ".join(eval_layouts(TUBE_LAYOUTS)))
    elif sys.argv[1] == "tube_start_layouts":
        print(" ".join(eval_layouts(TUBE_START_LAYOUTS)))
    elif sys.argv[1] == "small_layouts":
        print(" ".join(eval_layouts(SMALL_LAYOUTS)))
    elif sys.argv[1] == "choice_layouts":
        print(" ".join(eval_layouts(CHOICE_LAYOUTS)))
    elif sys.argv[1] == "overrides":
        print("\n".join(task_for(sys.argv[2]).overrides()))


if __name__ == "__main__":
    main()
