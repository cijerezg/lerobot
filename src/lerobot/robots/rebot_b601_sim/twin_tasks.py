"""Dense staged scorers of the twin eval layouts (tasks.py tiers twin_l1 .. twin_l4; scenes twin_l<L>_<config>_p<k>).

Same seam as scorer.py: ``poll()`` once per control tick advances the furthest stage (the episode ``reward`` written to
subtask_events.json stays the cumulative stage value, so reward.<layout> and the tier roll-ups are unchanged) and returns
the subtask cursor. On top of that every scorer keeps a DENSE value: stage value + within-stage progress toward the next
stage, in [stage value, next stage value), monotone non-decreasing over the episode (the running max is kept), plus the
raw quantities behind it. ``episode_extra()`` returns what the runtime adds to the episode record: ``reward_dense``, the
stage timeline, carried time, drop record (sim_summary.py rolls ``reward_dense.<layout>`` up next to ``reward.<layout>``
and writes the per-episode json).

Pick-and-place ladder (one object into the basket; ``pick_place_scorer(body, cursors)`` builds the class; the L4 sequence
tasks chain one ladder per object (``sequence_scorer``), equal shares of [0, 1] in task order, each ladder starting when the
previous object is placed):

| stage | value | predicate | progress toward it |
|---|---|---|---|
| approach | 0.1 | finger midpoint within APPROACH_M of the object's nearest collision geom (or touching it) | 1 - clip(d / d0) (d0 = distance at the first tick) |
| contact | 0.2 | a finger body touches the object | fraction of the last CONTACT_WINDOW ticks with contact |
| lifted | 0.4 | touched and the object centre LIFT_M above its rest height | clip(lift / LIFT_M) while touched |
| transport | 0.7 | lifted and the object centre within TRANSPORT_NEAR_M of the basket footprint | 1 - clip(d_basket / d_basket_at_lift) while lifted |
| over | 0.85 | object centre inside the basket footprint and above the rim | 1 - clip(d_basket / TRANSPORT_NEAR_M) |
| inside | 1.0 | inside the footprint, below the rim, no gripper contact, for INSIDE_HOLD_S | 0.5 x (below the rim, untouched) + 0.5 x hold fraction |

Drop: once the object has been lifted, the moment it is back on the table (lift < DROP_LIFT_M), or off the desk, or at rest for REST_S (lying on its side), untouched and outside the
footprint, a ``drop`` record stores the distance from the object centre to the basket footprint (how near the miss was)
and the dense value is frozen until the object is lifted again (the stage value never freezes: the ladder is the
stage walk of scorer.py). Footprint = the basket's collision-geom box in the basket frame; rim = its top.

Fold ladder (twin_l3_fold_kit; the kit's hinges give the angles: total fold angle of a flap = strip hinge + flap hinge):
flap_grasped 0.1 (a finger touches a flap or its hinge strip; progress 1 - d / d0 to the nearest flap collision geom),
inner_past_vertical 0.3 (bits flap, left, > FOLD_VERTICAL_DEG), inner_flat 0.5 (> FOLD_FLAT_DEG), outer_past_vertical 0.7
(card flap, right), closed 1.0 (both > FOLD_FLAT_DEG and the gripper clear of the kit); progress = the angle fraction of
the band. The order is the T067 script's (bits flap first); folding the other flap first still completes the ladder,
with the credit arriving late.
"""

from collections import deque

import mujoco
import numpy as np

from .easy_tasks import body_surface_distance
from .scorer import TaskScorer

BASKET = "basket"
APPROACH_M = 0.05  # finger midpoint to the nearest point of the object's collision geoms (the finger bodies sit 4.2 cm behind the tips): approach reached
CONTACT_WINDOW = 10  # control ticks; contact progress = fraction of the window with finger-object contact
LIFT_M = 0.03  # object centre above its settled rest height: lifted
TRANSPORT_NEAR_M = 0.10  # object centre to the basket footprint edge while lifted: transport reached
INSIDE_HOLD_S = 0.5  # sim seconds inside, below the rim and untouched before the placement counts
DROP_LIFT_M = 0.01  # lift below this after a lift = back on the table (drop check)
DESK_X = (-0.0445, 0.7175)  # twin desk top extents in the table frame (room.xml); outside = off the desk (the collider plane is infinite)
DESK_Y = (-0.61, 0.61)
REST_SPEED = 0.01  # m/s: an untouched object slower than this for REST_S after a lift has come to rest (lying on its side counts as dropped)
REST_S = 0.5
TIMELINE_STEP = 0.02  # dense increase that adds a timeline entry (stage changes and drops always do)
KIT = "bits_kit"
FLAPS = {"inner": ("bits_kit_left", "bits_kit_left_strip"), "outer": ("bits_kit_right", "bits_kit_right_strip")}  # flap body, hinge strip
HINGES = {"inner": ("bits_kit_hinge_left", "bits_kit_hinge_left_flap"), "outer": ("bits_kit_hinge_right", "bits_kit_hinge_right_flap")}
FOLD_VERTICAL_DEG = 90.0  # total fold angle of a flap past vertical
FOLD_FLAT_DEG = 170.0  # total fold angle of a flap lying on the stack (180 = flat)


class DenseScorer(TaskScorer):
    """TaskScorer plus the dense value: subclasses fill ``self.progress`` (next-stage name -> [0, 1]) and ``self.frozen`` in
    ``_predicates()`` and may extend ``extra()``."""

    def _reset(self) -> None:
        self.dense = 0.0
        self.progress: dict[str, float] = {}
        self.frozen = False
        self.timeline: list[dict] = []
        self._last_logged = -1.0

    def poll(self) -> int:
        st = self._predicates()
        self._advance(st)
        last = len(self.STAGES) - 1
        base = self.STAGES[self.stage][1] if self.stage >= 0 else 0.0
        if self.stage == last:
            dense = 1.0
        else:
            nxt_name, nxt_value, _ = self.STAGES[self.stage + 1]
            dense = base + float(np.clip(self.progress.get(nxt_name, 0.0), 0.0, 1.0)) * (nxt_value - base)
        self.dense = max(self.dense, base if self.frozen else dense)
        if self.dense - self._last_logged >= TIMELINE_STEP or (self.timeline and self.timeline[-1]["stage"] != self.stage_name):
            self.timeline.append(dict(t=round(self.t, 2), stage=self.stage_name, reward=self.reward, dense=round(self.dense, 4)))
            self._last_logged = self.dense
        want = self.STAGES[self.stage][2] if self.stage >= 0 else 0
        if want != self.cursor:
            self._log(want, f"stage {self.stage_name}")
        return self.cursor

    def extra(self) -> dict:
        return {}

    def episode_extra(self) -> dict:
        """Added to the episode record of subtask_events.json by the RTC env worker."""
        return _py(dict(reward_dense=round(self.dense, 4), dense_timeline=list(self.timeline), **self.extra()))


def _py(x):
    """numpy scalars/arrays -> plain Python for json.dumps (the raw predicate flags are numpy bools)."""
    if isinstance(x, dict): return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_py(v) for v in x]
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    return x


class PickPlaceLadder:
    """One object's pick-and-place quantities, predicates and progress; owned by a scorer, stateful over the episode."""

    NAMES = ("approach", "contact", "lifted", "transport", "over", "inside")

    def __init__(self, scorer: TaskScorer, body: str) -> None:
        self.s = scorer
        m = scorer.model
        self.body = m.body(body).id
        self.basket = m.body(BASKET).id
        # basket footprint and rim in the basket frame, from its collision geoms' local AABBs
        lo, hi = [], []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == self.basket and (m.geom_contype[g] or m.geom_conaffinity[g]):
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, m.geom_quat[g]); Rm = Rm.reshape(3, 3)
                cw = m.geom_pos[g] + Rm @ c; hw = np.abs(Rm) @ h
                lo.append(cw - hw); hi.append(cw + hw)
        self.foot_lo, self.foot_hi = np.min(lo, 0), np.max(hi, 0)
        self.reset()

    def reset(self) -> None:
        d = self.s.data
        self.z0 = float(d.xpos[self.body][2])
        self.d0 = None
        self.contacts = deque(maxlen=CONTACT_WINDOW)
        self.lifted_ever = False
        self.d_basket_at_lift = None
        self.carried_s = 0.0
        self.last_t = None
        self.drop = None
        self.drops = 0
        self.last_p = None
        self.rest_since = None
        self.basket_xy0 = None
        self.xy0 = None
        self.frozen = False
        self.inside_since = None
        self.placed = False
        self.raw: dict = {}

    def update(self) -> tuple[dict, dict]:
        s, d = self.s, self.s.data
        p = d.xpos[self.body]
        mid = s.midpoint()
        dist = body_surface_distance(s.model, d, self.body, mid)
        if self.d0 is None:
            self.d0 = max(dist, 1e-3)
        touching = s.touching(self.body)
        contact = bool(touching & set(s.tips))
        gripper_touch = bool(touching & s.gripper)
        self.contacts.append(contact)
        lift = float(p[2] - self.z0)
        lifted = contact and lift > LIFT_M
        # basket frame
        Rb = d.xmat[self.basket].reshape(3, 3)
        q = Rb.T @ (p - d.xpos[self.basket])
        in_footprint = bool(self.foot_lo[0] < q[0] < self.foot_hi[0] and self.foot_lo[1] < q[1] < self.foot_hi[1])
        d_basket = float(np.linalg.norm(np.maximum(0.0, np.maximum(self.foot_lo[:2] - q[:2], q[:2] - self.foot_hi[:2]))))
        rim_z = float(d.xpos[self.basket][2] + self.foot_hi[2])
        above_rim = p[2] > rim_z
        t = s.t
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        if lifted:
            if not self.lifted_ever or self.d_basket_at_lift is None:
                self.d_basket_at_lift = max(d_basket, 1e-3)
            self.lifted_ever = True
            self.carried_s += dt
            self.frozen = False
        off_desk = not (DESK_X[0] < p[0] < DESK_X[1] and DESK_Y[0] < p[1] < DESK_Y[1])
        speed = 0.0 if self.last_p is None or dt <= 0 else float(np.linalg.norm(p - self.last_p) / dt)
        self.last_p = p.copy()
        self.rest_since = None if (contact or speed > REST_SPEED) else (t if self.rest_since is None else self.rest_since)
        at_rest = self.rest_since is not None and t - self.rest_since >= REST_S
        if self.lifted_ever and not self.placed and not contact and (lift < DROP_LIFT_M or off_desk or at_rest) and not in_footprint and not self.frozen:
            self.frozen = True
            self.drops += 1
            if self.drop is None:
                self.drop = dict(t=round(t, 2), xy=[round(float(p[0]), 4), round(float(p[1]), 4)], distance_to_basket_m=round(d_basket, 4),
                                 distance_to_basket_centre_m=round(float(np.linalg.norm(q[:2])), 4), off_desk=off_desk)
        basket_moved = float(np.linalg.norm(d.xpos[self.basket][:2] - self.basket_xy0)) if self.basket_xy0 is not None else 0.0
        if self.basket_xy0 is None: self.basket_xy0 = d.xpos[self.basket][:2].copy()
        if self.xy0 is None: self.xy0 = p[:2].copy()
        displaced = float(np.linalg.norm(p[:2] - self.xy0))  # object moved from its start without being placed: knocked over / pushed
        settled_inside = in_footprint and not above_rim and not gripper_touch
        self.inside_since = None if not settled_inside else (t if self.inside_since is None else self.inside_since)
        hold = 0.0 if self.inside_since is None else t - self.inside_since
        inside = settled_inside and hold >= INSIDE_HOLD_S
        if inside:
            self.placed = True
        pred = dict(approach=dist < APPROACH_M or contact, contact=contact, lifted=lifted, transport=lifted and d_basket < TRANSPORT_NEAR_M,
                    over=in_footprint and above_rim, inside=inside)
        prog = dict(approach=1.0 - min(dist / self.d0, 1.0), contact=sum(self.contacts) / CONTACT_WINDOW,
                    lifted=min(max(lift, 0.0) / LIFT_M, 1.0) if contact else 0.0,
                    transport=(1.0 - min(d_basket / self.d_basket_at_lift, 1.0)) if (lifted and self.d_basket_at_lift) else 0.0,
                    over=1.0 - min(d_basket / TRANSPORT_NEAR_M, 1.0) if lifted else 0.0,
                    inside=0.5 * float(settled_inside) + 0.5 * min(hold / INSIDE_HOLD_S, 1.0))
        self.raw = dict(d_finger_surface_m=round(dist, 4), contact=contact, lift_m=round(lift, 4), d_basket_m=round(d_basket, 4), in_footprint=in_footprint,
                        above_rim=above_rim, carried_time_s=round(self.carried_s, 2), drops=self.drops, drop=self.drop, off_desk=off_desk, basket_moved_m=round(basket_moved, 4), displaced_m=round(displaced, 4))
        return pred, prog


def pick_place_scorer(body: str, cursors: tuple[int, ...]) -> type[DenseScorer]:
    """Scorer class for "put the <body> in the basket"; ``cursors`` = the script step each stage justifies (6 entries)."""
    values = (0.1, 0.2, 0.4, 0.7, 0.85, 1.0)

    class PickPlaceScorer(DenseScorer):
        STAGES = [(name, value, cursor) for name, value, cursor in zip(PickPlaceLadder.NAMES, values, cursors, strict=True)]
        BODY = body

        def _setup(self) -> None:
            self.ladder = PickPlaceLadder(self, body)

        def _reset(self) -> None:
            super()._reset()
            self.ladder.reset()

        def _predicates(self) -> dict[str, bool]:
            pred, self.progress = self.ladder.update()
            self.frozen = self.ladder.frozen
            return pred

        def extra(self) -> dict:
            return dict(carried_time_s=round(self.ladder.carried_s, 2), drop=self.ladder.drop, drops=self.ladder.drops, raw=self.ladder.raw)

    PickPlaceScorer.__name__ = f"PickPlace_{body}"
    return PickPlaceScorer


def sequence_scorer(bodies: tuple[str, ...], cursors: tuple[tuple[int, ...], ...]) -> type[DenseScorer]:
    """"Put the <a>, the <b> ... in the basket": one pick-and-place ladder per object in task order, the i-th scaled to
    [i / n, (i + 1) / n]; a ladder only runs once the previous object is placed (its carried time, drops and progress are
    not counted before its turn). ``cursors[i]`` = the script step each of the i-th object's six stages justifies."""
    values = (0.1, 0.2, 0.4, 0.7, 0.85, 1.0)
    n = len(bodies)
    assert n == len(cursors) >= 2

    class SequenceScorer(DenseScorer):
        STAGES = [(f"{body}_{name}", round((i + v) / n, 4), c) for i, body in enumerate(bodies)
                  for name, v, c in zip(PickPlaceLadder.NAMES, values, cursors[i], strict=True)]
        BODIES = bodies

        def _setup(self) -> None:
            self.ladders = [PickPlaceLadder(self, body) for body in bodies]

        def _reset(self) -> None:
            super()._reset()
            for ladder in self.ladders:
                ladder.reset()

        def _predicates(self) -> dict[str, bool]:
            pred, prog = {}, {}
            per_object = len(PickPlaceLadder.NAMES)
            for i, (body, ladder) in enumerate(zip(bodies, self.ladders, strict=True)):
                active = i == 0 or self.ladders[i - 1].placed or self.stage >= i * per_object - 1
                if active:
                    p, g = ladder.update()
                    self.frozen = ladder.frozen
                else:
                    ladder.last_t = self.t  # no carried time before its turn
                    p = {name: False for name in PickPlaceLadder.NAMES}; g = {name: 0.0 for name in PickPlaceLadder.NAMES}
                pred |= {f"{body}_{name}": p[name] for name in p}
                prog |= {f"{body}_{name}": g[name] for name in g}
            self.progress = prog
            return pred

        def extra(self) -> dict:
            objects = {body: dict(carried_time_s=round(l.carried_s, 2), drop=l.drop, drops=l.drops, placed=l.placed, raw=l.raw)
                       for body, l in zip(bodies, self.ladders, strict=True)}
            return dict(carried_time_s=round(sum(l.carried_s for l in self.ladders), 2), objects=objects,
                        drop=next((l.drop for l in self.ladders if l.drop), None), drops=sum(l.drops for l in self.ladders))

    SequenceScorer.__name__ = "Sequence_" + "_".join(bodies)
    return SequenceScorer


class FoldKitScorer(DenseScorer):
    """Close the bit kit: bits flap (inner, left) over the middle, card flap (outer, right) over it."""

    STAGES = [("flap_grasped", 0.1, 0), ("inner_past_vertical", 0.3, 1), ("inner_flat", 0.5, 2), ("outer_past_vertical", 0.7, 3), ("closed", 1.0, 5)]

    def _setup(self) -> None:
        m = self.model
        self.kit_bodies = {m.body(KIT).id} | {m.body(n).id for pair in FLAPS.values() for n in pair}
        self.flap_bodies = {m.body(n).id for pair in FLAPS.values() for n in pair}
        self.qadr = {side: [int(m.jnt_qposadr[m.joint(j).id]) for j in joints] for side, joints in HINGES.items()}

    def _reset(self) -> None:
        super()._reset()
        self.d0 = None
        self.raw = {}

    def angles_deg(self) -> dict[str, float]:
        return {side: float(np.degrees(sum(self.data.qpos[a] for a in adr))) for side, adr in self.qadr.items()}

    def _predicates(self) -> dict[str, bool]:
        ang = self.angles_deg()
        mid = self.midpoint()
        dist = min(body_surface_distance(self.model, self.data, b, mid) for b in self.flap_bodies)
        if self.d0 is None:
            self.d0 = max(dist, 1e-3)
        touching = set()
        for b in self.flap_bodies:
            touching |= self.touching(b)
        grasped = bool(touching & set(self.tips))
        kit_touch = set()
        for b in self.kit_bodies:
            kit_touch |= self.touching(b)
        clear = not (kit_touch & self.gripper)
        inner, outer = ang["inner"], ang["outer"]
        band = FOLD_FLAT_DEG - FOLD_VERTICAL_DEG
        self.progress = dict(flap_grasped=1.0 - min(dist / self.d0, 1.0), inner_past_vertical=min(max(inner, 0.0) / FOLD_VERTICAL_DEG, 1.0),
                             inner_flat=min(max(inner - FOLD_VERTICAL_DEG, 0.0) / band, 1.0), outer_past_vertical=min(max(outer, 0.0) / FOLD_VERTICAL_DEG, 1.0),
                             closed=min(max(outer - FOLD_VERTICAL_DEG, 0.0) / band, 1.0))
        self.raw = dict(inner_deg=round(inner, 1), outer_deg=round(outer, 1), d_flap_m=round(dist, 4), grasped=grasped, gripper_clear=clear)
        return dict(flap_grasped=grasped, inner_past_vertical=inner > FOLD_VERTICAL_DEG, inner_flat=inner > FOLD_FLAT_DEG,
                    outer_past_vertical=outer > FOLD_VERTICAL_DEG, closed=inner > FOLD_FLAT_DEG and outer > FOLD_FLAT_DEG and clear)

    def extra(self) -> dict:
        return dict(raw=self.raw, angles_deg=self.angles_deg())
