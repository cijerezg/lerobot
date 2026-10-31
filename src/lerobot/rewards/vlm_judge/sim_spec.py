"""TaskSpec of a sim eval layout, built from robots/rebot_b601_sim/tasks.py so it cannot drift from what the policy and the
scorer use: ``script`` = the layout's eval_subtasks, subtasks = the script steps (criterion = assigner.step_criterion, contact
and precision from the Task) plus one off-task entry, stages = the scorer's STAGES (names, values, cursors) with an observable
criterion per stage name. ``TaskSpec.load("sim:<layout>")`` lands here; compare_to_scorer then matches the scorer's stage names.

Old layouts keep their hand-written JSON (bottle_basket, close_lid, nail_clippers); the twin ones (pick-place ladder, L4
sequences, fold kit) are generated. The sim package is imported lazily (mujoco)."""

from .assigner import step_criterion
from .task_spec import Stage, Subtask, TaskSpec

PICK_PLACE = {  # twin_tasks.PickPlaceLadder.NAMES, thresholds from twin_tasks.py
    "approach": "the fingertips within about 5 cm of {obj}",
    "contact": "a finger touches {obj}",
    "lifted": "{obj} held by the fingers and more than 3 cm above where it rested",
    "transport": "{obj} lifted and within about 10 cm of the basket",
    "over": "{obj} above the basket opening, higher than the rim",
    "inside": "{obj} inside the basket below the rim, no finger touching it",
}
FOLD = {  # twin_tasks.FoldKitScorer (bits flap = inner/left, card flap = outer/right)
    "flap_grasped": "a finger touches one of the kit's side flaps or its hinge strip",
    "inner_past_vertical": "the flap with the bits rotated past vertical toward the middle panel",
    "inner_flat": "the flap with the bits lying flat on the middle panel",
    "outer_past_vertical": "the flap with the white card rotated past vertical toward the middle",
    "closed": "both flaps folded flat on the kit and the gripper clear of it",
}
OLD = {"BottleBasketScorer": "bottle_basket", "CloseLidScorer": "close_lid", "NailTaskScorer": "nail_clippers"}


def spec_for_layout(layout: str) -> TaskSpec:
    from lerobot.robots.rebot_b601_sim import tasks as sim_tasks

    name = sim_tasks.layout_name(layout)
    task = sim_tasks.TASKS[name]
    scorer = task.scorer
    if scorer.__name__ in OLD:
        spec = TaskSpec.load(OLD[scorer.__name__])
        assert list(spec.script) == list(task.eval_subtasks), f"{name}: {OLD[scorer.__name__]}.json script differs from tasks.py"
        return spec

    def noun(body: str) -> str:
        return sim_tasks._TWIN_BASKET[body][0]

    if hasattr(scorer, "BODIES"):  # L4 sequence: '<body>_<ladder stage>'
        n = len(scorer.BODIES)
        crit = {}
        for i, body in enumerate(scorer.BODIES):
            for stage, text in PICK_PLACE.items():
                crit[f"{body}_{stage}"] = f"object {i + 1} of {n}: " + text.format(obj=noun(body))
    elif hasattr(scorer, "BODY"):
        crit = {stage: text.format(obj=noun(scorer.BODY)) for stage, text in PICK_PLACE.items()}
    else:
        crit = FOLD

    script = list(task.eval_subtasks)
    subtasks = [Subtask(id=i, text=s, cursor=i, contact=task.eval_subtask_contacts[i], precision=task.eval_subtask_precisions[i],
                        criterion=step_criterion(s)) for i, s in enumerate(script)]
    subtasks.append(Subtask(id=len(script), text="idle or off-task", cursor=-1, contact="na", precision=1,
                            criterion="arm not moving for the whole strip, or doing something none of the entries above describes"))
    stages = [Stage(id=0, name="none", value=0.0, cursor=0, criterion="nothing below holds")]
    stages += [Stage(id=i + 1, name=s, value=float(v), cursor=c, criterion=crit[s]) for i, (s, v, c) in enumerate(scorer.STAGES)]
    spec = TaskSpec(name=f"sim:{name}", task=task.task, script=tuple(script), subtasks=tuple(subtasks), stages=tuple(stages))
    spec.validate()
    return spec
