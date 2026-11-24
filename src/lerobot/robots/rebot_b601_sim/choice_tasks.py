"""Scorer of the object-choice layouts (tasks.py tier ``choice``; scenes choice_<pair>_ask_<object>_p<k>,
scripts/sim_builders/object_choice): two objects and the basket, the task names one object.

The reward is the pick-and-place ladder of the NAMED object (twin_tasks.py). On top of it the episode record gets ``choice``:
for both objects the first time the fingers came within APPROACH_M, touched it, lifted it, the smallest finger distance and
whether it ended in the basket, and which object was approached / touched / lifted FIRST (None = neither).
"""

from .twin_tasks import DenseScorer, PickPlaceLadder, pick_place_scorer

EVENTS = ("approach", "contact", "lifted")


def choice_scorer(asked: str, other: str, cursors: tuple[int, ...]) -> type[DenseScorer]:
    class ChoiceScorer(pick_place_scorer(asked, cursors)):
        def _setup(self) -> None:
            super()._setup()
            self.other = PickPlaceLadder(self, other)

        def _reset(self) -> None:
            super()._reset()
            self.other.reset()
            self.seen = {body: dict(start_distance_m=None, min_distance_m=None, **{event: None for event in EVENTS}) for body in (asked, other)}

        def _predicates(self) -> dict[str, bool]:
            pred = super()._predicates()
            for body, ladder, p in ((asked, self.ladder, pred), (other, self.other, self.other.update()[0])):
                seen, dist = self.seen[body], ladder.raw["d_finger_surface_m"]
                if seen["start_distance_m"] is None:
                    seen["start_distance_m"] = dist
                seen["min_distance_m"] = dist if seen["min_distance_m"] is None else min(seen["min_distance_m"], dist)
                for event in EVENTS:
                    if p[event] and seen[event] is None:
                        seen[event] = round(self.t, 2)
            return pred

        def first(self, event: str) -> str | None:
            times = {body: seen[event] for body, seen in self.seen.items() if seen[event] is not None}
            return min(times, key=times.get) if times else None

        def extra(self) -> dict:
            objects = {body: dict(self.seen[body], placed=ladder.placed, displaced_m=ladder.raw.get("displaced_m"))
                       for body, ladder in ((asked, self.ladder), (other, self.other))}
            return dict(super().extra(), choice=dict(asked=asked, other=other, objects=objects, **{f"first_{event}": self.first(event) for event in EVENTS}))

    ChoiceScorer.__name__ = f"Choice_{asked}_not_{other}"
    return ChoiceScorer
