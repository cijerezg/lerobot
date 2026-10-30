"""Scorers of the easy-tier scenes (tasks.py): bottle_basket, lift_bottle, close_lid.

Body names are the scene contract: ``bottle`` (free body, collision geom + visual meshes), ``basket`` (collision
geoms give the footprint), ``headphone_case`` + ``headphone_lid`` with hinge ``lid_hinge`` (0 = open; the closed
angle is the nonzero end of the joint's ``range``). "Held" is both finger bodies in contact with the object
(scorer.pinched), so it does not depend on the gripper reading; "reach" is the finger midpoint within a few cm of
the object's centre, except close_lid: the lid is grasped at its rim (the demos rim-pinch the side wall near the far
end, 9-10 cm from the lid's centre of mass), so its reach is the finger midpoint within LID_REACH_M of the nearest
point of any lid collision geom (the tray floor and the four 7 mm rim walls). The finger bodies sit 4.2 cm behind
the tips, so a pinch 1.5 cm deep on a wall puts the midpoint 2.7 cm from the rim. Thresholds are plain numbers below.
"""

import numpy as np

from .scorer import TaskScorer

BOTTLE = "bottle"
BASKET = "basket"
LID = "headphone_lid"
CASE = "headphone_case"
LID_HINGE = "lid_hinge"
REACH_M = 0.06  # finger midpoint to the bottle centre (bottle radius 27.5 mm)
BASKET_LIFT_M = 0.03  # bottle centre above its rest height
LIFT_M = 0.05  # lift_bottle: bottle centre above its rest height
HOLD_S = 2.0  # lift_bottle: held and lifted continuously, sim seconds
LID_REACH_M = 0.05  # finger midpoint to the nearest point of a lid collision geom (2026-10-28; was 8 cm to the lid COM, which a rim pinch misses)
LID_CLOSED_TOL = 0.08  # fraction of the closed angle (close_lid: pi, soft stop; its XML scores closed as > 2.9 rad)


class BottleBasketScorer(TaskScorer):
    STAGES = [("reach", 0.1, 0), ("held", 0.3, 1), ("lifted", 0.5, 1), ("over", 0.8, 2), ("inside", 1.0, 3)]

    def _setup(self) -> None:
        self.bottle = self.model.body(BOTTLE).id
        self.basket = self.model.body(BASKET).id

    def _reset(self) -> None:
        self.z0 = float(self.data.xpos[self.bottle][2])

    def _predicates(self) -> dict[str, bool]:
        p = self.data.xpos[self.bottle]
        lo, hi = self.aabb(self.basket)
        in_footprint = bool(lo[0] < p[0] < hi[0] and lo[1] < p[1] < hi[1])
        touching = self.touching(self.bottle)
        held = set(self.tips) <= touching
        lifted = p[2] > self.z0 + BASKET_LIFT_M
        return dict(
            reach=np.linalg.norm(p - self.midpoint()) < REACH_M,
            held=held,
            lifted=held and lifted,
            over=lifted and in_footprint,
            inside=in_footprint and p[2] < hi[2] and not (touching & self.gripper),
        )


class LiftBottleScorer(TaskScorer):
    STAGES = [("reach", 0.2, 0), ("held", 0.5, 1), ("lifted", 0.8, 1), ("held_lifted", 1.0, 1)]

    def _setup(self) -> None:
        self.bottle = self.model.body(BOTTLE).id

    def _reset(self) -> None:
        self.z0 = float(self.data.xpos[self.bottle][2])
        self.lift_since = None

    def _predicates(self) -> dict[str, bool]:
        p = self.data.xpos[self.bottle]
        held = self.pinched(self.bottle)
        lifted = held and p[2] > self.z0 + LIFT_M
        self.lift_since = None if not lifted else (self.t if self.lift_since is None else self.lift_since)
        return dict(
            reach=np.linalg.norm(p - self.midpoint()) < REACH_M,
            held=held,
            lifted=lifted,
            held_lifted=lifted and self.t - self.lift_since >= HOLD_S,
        )


def body_surface_distance(model, data, body: int, point: np.ndarray) -> float:
    """Distance from ``point`` to the nearest collision geom of ``body``: each geom's local AABB (exact for boxes) clamped
    in the geom frame."""
    best = np.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != body or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        local = data.geom_xmat[g].reshape(3, 3).T @ (point - data.geom_xpos[g])
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        best = min(best, float(np.linalg.norm(local - np.clip(local, c - h, c + h))))
    return best


class CloseLidScorer(TaskScorer):
    STAGES = [("reach", 0.1, 0), ("quarter", 0.4, 1), ("three_quarters", 0.7, 1), ("closed", 1.0, 2)]

    def _setup(self) -> None:
        m = self.model
        self.lid = m.body(LID).id
        self.case = m.body(CASE).id
        j = m.joint(LID_HINGE).id
        self.qadr = int(m.jnt_qposadr[j])
        lo, hi = m.jnt_range[j]
        self.closed = float(hi if lo == 0 else lo)

    def _predicates(self) -> dict[str, bool]:
        frac = float(self.data.qpos[self.qadr]) / self.closed
        clear = not ((self.touching(self.lid) | self.touching(self.case)) & self.gripper)
        return dict(
            reach=body_surface_distance(self.model, self.data, self.lid, self.midpoint()) < LID_REACH_M,
            quarter=frac >= 0.25,
            three_quarters=frac >= 0.75,
            closed=frac >= 1.0 - LID_CLOSED_TOL and clear,
        )
