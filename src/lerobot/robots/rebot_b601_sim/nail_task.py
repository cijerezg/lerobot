"""Nail-clipper task (T085) scorer: subtask cursor and staged reward from the MuJoCo state.

On the real rig the operator walks ``cfg.policy.eval_subtasks`` with the console keys; in the sim nobody
is watching, so ``NailTaskScorer.poll()`` returns the cursor the body poses justify and the RTC env worker
syncs ``RTCSharedState`` to it (same seam as the console: ``advance_subtask``). One completion predicate
per step, forward only, plus one back rule: a clipper dropped while moving or releasing it sends the
cursor back to its grasp step. Both clippers seated = success (``SUCCESS``, scorer.py).

Script (tasks.py, T085): 0 grasp large, 1 move large, 2 release large, 3 push large, 4 grasp small,
5 move small, 6 release small, 7 push small, 8 return to home (never auto-advanced).

Stages (reward): reach large 0.1, large held 0.2, large over the case 0.3, large seated 0.5, small held 0.6,
small over the case 0.7, both seated 1.0 (the stage walk is sequential, scorer.py).

Geometry: slot footprints in the case body frame from the scan height map (migration/real2sim_2026-10-17
object_materials.py: slot = mesh height < 13 mm; centres/extents of the two components; floor median
7.6 / 8.3 mm + the 2.5 mm mesh shift). Thresholds are plain numbers below.
"""

import numpy as np

from .scorer import SUCCESS, TaskScorer  # noqa: F401  (SUCCESS re-exported for older imports)

SLOTS = {
    "clipper_large": dict(center=(-0.0156, -0.0039), half=(0.0095, 0.037), floor_z=0.0101),
    "clipper_small": dict(center=(0.0160, -0.0049), half=(0.0085, 0.025), floor_z=0.0108),
}
CASE = "clipper_case"
GRIPPER_CLOSED_DEG = -60.0  # dataset gripper units: closed on a clipper about -32, open -110..-140
REACH_DIST_M = 0.04  # clipper centre to finger midpoint
HELD_DIST_M = 0.035  # clipper centre to finger midpoint
LIFT_Z_M = 0.015  # clipper centre above the table (rest 4.6-5.1 mm)
NEAR_SLOT_M = 0.06  # horizontal, case frame: over the case (88 x 119 mm)
NEAR_SLOT_Z_M = 0.08
RIM_Z_M = 0.025  # case frame; the rim top is 21 mm
SEAT_TOL_M = 0.003
# step -> (clipper, predicate name); predicate = what must hold to leave the step
STEPS = [
    ("clipper_large", "held"), ("clipper_large", "near"), ("clipper_large", "released"), ("clipper_large", "seated"),
    ("clipper_small", "held"), ("clipper_small", "near"), ("clipper_small", "released"), ("clipper_small", "seated"),
]
GRASP_STEP = {"clipper_large": 0, "clipper_small": 4}


class NailTaskScorer(TaskScorer):
    STAGES = [
        ("reach_large", 0.1, 0), ("held_large", 0.2, 1), ("near_large", 0.3, 2), ("seated_large", 0.5, 4),
        ("held_small", 0.6, 5), ("near_small", 0.7, 6), ("seated_both", 1.0, 8),
    ]

    def _setup(self) -> None:
        model = self.model
        self.case = model.body(CASE).id
        self.bodies = {name: model.body(name).id for name in SLOTS}
        self.half_h = {name: float(model.geom_size[[g for g in range(model.ngeom) if model.geom_bodyid[g] == b and model.geom_contype[g]][0]][2]) for name, b in self.bodies.items()}

    def _state(self, name: str) -> dict:
        d = self.data; p = d.xpos[self.bodies[name]]; mid = self.midpoint()
        q = d.xmat[self.case].reshape(3, 3).T @ (p - d.xpos[self.case])  # clipper centre in the case frame
        slot = SLOTS[name]; dx, dy = abs(q[0] - slot["center"][0]), abs(q[1] - slot["center"][1])
        closed = self.gripper_deg() > GRIPPER_CLOSED_DEG
        dist = np.linalg.norm(p - mid)
        held = closed and dist < HELD_DIST_M and p[2] > LIFT_Z_M
        in_footprint = dx < slot["half"][0] + 0.004 and dy < slot["half"][1]
        return dict(
            reach=dist < REACH_DIST_M,
            held=held,
            near=held and np.hypot(q[0] - slot["center"][0], q[1] - slot["center"][1]) < NEAR_SLOT_M and q[2] < NEAR_SLOT_Z_M,
            in_slot=in_footprint and q[2] < RIM_Z_M,
            released=in_footprint and q[2] < RIM_Z_M and not closed,
            seated=dx < slot["half"][0] and dy < slot["half"][1] and q[2] < slot["floor_z"] + self.half_h[name] + SEAT_TOL_M,
        )

    def _predicates(self) -> dict[str, bool]:
        large, small = self._state("clipper_large"), self._state("clipper_small")
        return dict(reach_large=large["reach"], held_large=large["held"], near_large=large["near"], seated_large=large["seated"],
                    held_small=small["held"], near_small=small["near"], seated_both=large["seated"] and small["seated"])

    def poll(self) -> int:
        """Advance the cursor by the step predicates (forward, drop-back) and the stage by the reward predicates."""
        if self.cursor < len(STEPS):
            name, pred = STEPS[self.cursor]; st = self._state(name)
            if st[pred]:
                self._log(self.cursor + 1, f"{name} {pred}")
            elif pred in ("near", "released") and not st["held"] and not st["in_slot"]:
                self._log(GRASP_STEP[name], f"{name} dropped")
        self._advance(self._predicates())
        return self.cursor
