"""Staged task scorer for the sim eval: one interface per task (tasks.py maps scene -> scorer).

A scorer owns an ordered stage list ``STAGES = [(predicate, value, cursor), ...]``; predicates come from
``_predicates()`` (body poses, contacts, joint angles of the MuJoCo state). ``poll()`` is called once per
control tick by the RTC env worker: it advances the furthest stage reached (sequentially, the last stage =
success is reachable directly), so the episode's ``reward`` is the value of the furthest stage and never
decreases; it returns the subtask cursor the shared state should hold (``cursor`` of the furthest stage;
the nail task overrides this with its forward/drop-back walk). Reaching the last stage sets ``SUCCESS``,
which sim_autostart reports as TeleopEvents.SUCCESS so ``terminate_on_success`` ends the episode.

Events (``events``, written to subtask_events.json by the env worker): ``t`` = sim seconds since the scene
reset (the videos' and rerun's clock), ``cursor``, ``stage`` (name), ``reward``, ``why``.
"""

import threading

import mujoco
import numpy as np

# The task's last stage holds; sim_autostart reports it as TeleopEvents.SUCCESS. Cleared on scene reset.
SUCCESS = threading.Event()
GRIPPER_BODIES = ("gripper_end", "gripper_left", "gripper_right")


class TaskScorer:
    STAGES: list[tuple[str, float, int]] = []  # (predicate name, cumulative reward, subtask cursor it justifies)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, gripper_deg) -> None:
        """``gripper_deg()`` returns the present gripper reading in dataset units (0 closed, -238.8 open)."""
        self.model, self.data, self.gripper_deg = model, data, gripper_deg
        self.tips = [model.body("gripper_left").id, model.body("gripper_right").id]
        self.gripper = {model.body(n).id for n in GRIPPER_BODIES}
        self._setup()
        self.reset()

    def _setup(self) -> None:
        pass

    def _reset(self) -> None:
        pass

    def reset(self) -> None:
        self.cursor = 0
        self.stage = -1
        self.reward = 0.0
        self.events: list[dict] = []
        self.t0 = float(self.data.time)
        SUCCESS.clear()
        self._reset()

    @property
    def stage_name(self) -> str | None:
        return None if self.stage < 0 else self.STAGES[self.stage][0]

    @property
    def t(self) -> float:
        return float(self.data.time) - self.t0

    def _predicates(self) -> dict[str, bool]:
        raise NotImplementedError

    def poll(self) -> int:
        """Advance the stage and the cursor from the predicates; returns the cursor the shared state should hold."""
        st = self._predicates()
        self._advance(st)
        want = self.STAGES[self.stage][2] if self.stage >= 0 else 0
        if want != self.cursor:
            self._log(want, f"stage {self.stage_name}")
        return self.cursor

    def _advance(self, st: dict[str, bool]) -> None:
        last = len(self.STAGES) - 1
        while self.stage < last and st[self.STAGES[self.stage + 1][0]]:
            self._reach(self.stage + 1)
        if st[self.STAGES[last][0]] and self.stage < last:
            self._reach(last)
        if self.stage == last and not SUCCESS.is_set():
            SUCCESS.set()

    def _reach(self, stage: int) -> None:
        self.stage = stage
        self.reward = self.STAGES[stage][1]
        self._log(self.cursor, f"reached {self.STAGES[stage][0]}")

    def _log(self, cursor: int, why: str) -> None:
        self.cursor = cursor
        self.events.append(dict(t=round(self.t, 2), cursor=cursor, stage=self.stage_name, reward=self.reward, why=why))

    # --- geometry helpers ---------------------------------------------------------------------------------------

    def midpoint(self) -> np.ndarray:
        return (self.data.xpos[self.tips[0]] + self.data.xpos[self.tips[1]]) / 2

    def touching(self, body: int) -> set[int]:
        """Bodies in contact with ``body`` this tick."""
        d, m = self.data, self.model
        out = set()
        for c in d.contact[: d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b1 == body:
                out.add(b2)
            elif b2 == body:
                out.add(b1)
        return out

    def pinched(self, body: int) -> bool:
        return set(self.tips) <= self.touching(body)

    def aabb(self, body: int) -> tuple[np.ndarray, np.ndarray]:
        """World AABB (lo, hi) over the body's collision geoms, from each geom's local AABB corners."""
        m, d = self.model, self.data
        corners = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            local = c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners.append(local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        pts = np.concatenate(corners)
        return pts.min(axis=0), pts.max(axis=0)
