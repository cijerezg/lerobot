"""Dense staged scorer of "put the tube in the peg" (2026-11-15; bodies ``tube`` / ``tube_holder``, then ``tube_2`` /
``tube_holder_2``, ... of scripts/sim_builders/tube_peg/tube_peg.py: tube origin at its bottom centre, holder origin at the
plate's bottom centre, peg along the holder's z at the plate centre). Tube i belongs on holder i.

Ladder of one tube (value, script step it justifies; the script has four steps per tube and "return home" last):
    approach 0.10 (grasp)   finger midpoint within APPROACH_M of the tube
    contact  0.20 (grasp)   a gripper body touches the tube
    lifted   0.40 (move)    tube bottom LIFT_M above its rest height while touched
    at_peg   0.70 (insert)  tube axis within NEAR_M of the peg axis, tube bottom above the plate
    on_peg   0.85 (insert)  tube axis inside the bore radius of the peg axis, bottom ON_PEG_DEPTH_M below the peg top, upright
    inserted 1.00 (next)    on_peg and untouched for HOLD_S
With n tubes the stages are the n ladders in script order (names ``<stage>_<i>``, values (i + value) / n). Each tube's ladder
latches on its own, so a tube inserted out of order is credited once the earlier tubes are done.
Dense progress toward the next stage: distance to the tube (approach), lift height (lifted), horizontal distance to the peg axis
(at_peg), depth below the peg top (on_peg), untouched time (inserted).
"""

import numpy as np

from .easy_tasks import body_surface_distance
from .scorer import TaskScorer
from .twin_tasks import APPROACH_M, DenseScorer

NAMES = ("approach", "contact", "lifted", "at_peg", "on_peg", "inserted")
VALUES = (0.1, 0.2, 0.4, 0.7, 0.85, 1.0)
CURSORS = (1, 1, 2, 3, 3, 4)  # script step inside the tube's four, after the stage
LIFT_M = 0.02
NEAR_M = 0.015
BORE_R_M = 0.0038  # tube inner radius: the peg axis is inside the bore
ON_PEG_DEPTH_M = 0.003
UPRIGHT_COS = 0.9
HOLD_S = 0.5
FAR_M = 0.30  # approach / at_peg progress is 0 at this distance


class TubeLadder:
    """One tube and its holder: ``level`` = number of ladder stages reached so far (latched, in order)."""

    def __init__(self, scorer: TaskScorer, tube: str, holder: str) -> None:
        self.s = scorer
        m = scorer.model
        self.tube, self.holder = m.body(tube).id, m.body(holder).id
        peg = m.geom(f"{holder}_peg").id  # capsule along the holder's z
        self.peg_top = float(m.geom_pos[peg][2] + m.geom_size[peg][1] + m.geom_size[peg][0])
        plate = m.geom(f"{holder}_plate").id
        self.plate_top = float(m.geom_pos[plate][2] + m.geom_size[plate][2])

    def reset(self) -> None:
        self.level = 0
        self.rest_z = float(self.s.data.xpos[self.tube][2])
        self.untouched_since: float | None = None
        self.raw: dict = {}

    def update(self) -> dict[str, float]:
        """Advance ``level``; returns the progress toward each stage."""
        s, d = self.s, self.s.data
        touched = bool(s.gripper & s.touching(self.tube))
        reach = body_surface_distance(s.model, d, self.tube, s.midpoint())
        lift = float(d.xpos[self.tube][2]) - self.rest_z
        rel = d.xmat[self.holder].reshape(3, 3).T @ (d.xpos[self.tube] - d.xpos[self.holder])  # tube bottom centre in the holder frame
        off = float(np.linalg.norm(rel[:2]))
        depth = self.peg_top - float(rel[2])  # tube bottom below the peg top
        upright = float(d.xmat[self.tube].reshape(3, 3)[:, 2] @ d.xmat[self.holder].reshape(3, 3)[:, 2])
        on_peg = off < BORE_R_M and depth > ON_PEG_DEPTH_M and upright > UPRIGHT_COS
        if on_peg and not touched:
            self.untouched_since = s.t if self.untouched_since is None else self.untouched_since
        else:
            self.untouched_since = None
        held = 0.0 if self.untouched_since is None else s.t - self.untouched_since
        pred = (reach < APPROACH_M, touched, touched and lift > LIFT_M, off < NEAR_M and float(rel[2]) > self.plate_top, on_peg, held >= HOLD_S)
        while self.level < len(NAMES) and pred[self.level]:
            self.level += 1
        if pred[-1]:
            self.level = len(NAMES)
        self.raw = dict(level=self.level, peg_offset_mm=round(off * 1e3, 1), depth_mm=round(depth * 1e3, 1), lift_mm=round(lift * 1e3, 1), upright=round(upright, 3), touched=touched)
        return dict(
            approach=1.0 - (reach - APPROACH_M) / (FAR_M - APPROACH_M),
            contact=0.0,
            lifted=lift / LIFT_M if touched else 0.0,
            at_peg=1.0 - (off - NEAR_M) / (FAR_M - NEAR_M),
            on_peg=depth / ON_PEG_DEPTH_M if off < BORE_R_M else 0.0,
            inserted=held / HOLD_S,
        )


def tube_peg_scorer(n: int) -> type[DenseScorer]:
    """Scorer class for n tubes (bodies ``tube``, ``tube_2``, ...) on their holders, in script order."""
    suffixes = [""] + [f"_{i}" for i in range(2, n + 1)]

    class TubePegScorer(DenseScorer):
        STAGES = [(name if n == 1 else f"{name}_{i + 1}", round((i + value) / n, 4), 4 * i + cursor)
                  for i in range(n) for name, value, cursor in zip(NAMES, VALUES, CURSORS, strict=True)]

        def _setup(self) -> None:
            self.ladders = [TubeLadder(self, f"tube{sfx}", f"tube_holder{sfx}") for sfx in suffixes]

        def _reset(self) -> None:
            super()._reset()
            for ladder in self.ladders:
                ladder.reset()

        def _predicates(self) -> dict[str, bool]:
            pred, self.progress = {}, {}
            for i, ladder in enumerate(self.ladders):
                progress = ladder.update()
                for k, name in enumerate(NAMES):
                    stage = self.STAGES[len(NAMES) * i + k][0]
                    pred[stage] = ladder.level > k
                    self.progress[stage] = progress[name]
            last = self.STAGES[-1][0]  # the scorer accepts the last stage directly: it needs every tube inserted
            pred[last] = all(ladder.level == len(NAMES) for ladder in self.ladders)
            return pred

        def extra(self) -> dict:
            return dict(raw=[ladder.raw for ladder in self.ladders])

    TubePegScorer.__name__ = f"TubePeg_{n}"
    return TubePegScorer
