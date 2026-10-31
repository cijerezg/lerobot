"""Rate-limited judge for a live eval loop: ``observe`` every control step, one query in flight at a time on a worker
thread (a 31B model answers in seconds; the 30 Hz loop must not wait), ``latest`` is the newest judgement.

    runner = JudgeRunner(cfg, TaskSpec.load("bottle_basket"), JudgeClient(cfg))
    for step in range(...):
        runner.observe(step, t, {"top": top_rgb, "wrist": wrist_rgb})   # keeps the last frames_per_camera frames per camera
        j = runner.latest                                              # None until the first answer
    runner.close()
"""

import logging
import threading
from collections import deque

import numpy as np

from .client import JudgeClient
from .config import JudgeConfig
from .parse import Judgement
from .prompt import Frame
from .task_spec import TaskSpec

logger = logging.getLogger(__name__)


class JudgeRunner:
    def __init__(self, cfg: JudgeConfig, spec: TaskSpec, client: JudgeClient) -> None:
        self.cfg, self.spec, self.client = cfg, spec, client
        self.buffers: dict[str, deque[Frame]] = {c: deque(maxlen=cfg.frames_per_camera) for c in cfg.cameras}
        self.last_kept_t: dict[str, float] = {}
        self.latest: Judgement | None = None
        self.history: list[dict] = []  # every answer: t, judgement, meta
        self.progress_max = 0.0
        self.thread: threading.Thread | None = None

    def observe(self, step: int, t: float, images: dict[str, np.ndarray]) -> None:
        for cam in self.cfg.cameras:
            if cam in images and t - self.last_kept_t.get(cam, -1e9) >= self.cfg.frame_period_s - 1e-6:
                self.buffers[cam].append(Frame(t=t, camera=cam, image=images[cam]))
                self.last_kept_t[cam] = t
        if step % self.cfg.every_k_steps == 0:
            self.maybe_query(t)

    def maybe_query(self, t: float) -> bool:
        """Start a query on the buffered frames unless one is still running; returns whether one started."""
        if self.thread is not None and self.thread.is_alive():
            return False
        frames = [f for buf in self.buffers.values() for f in buf]
        if not frames:
            return False
        self.thread = threading.Thread(target=self._query, args=(t, frames), daemon=True)
        self.thread.start()
        return True

    def _query(self, t: float, frames: list[Frame]) -> None:
        try:
            judgement, meta = self.client.query(self.spec, frames)
        except Exception as e:  # a failed query is logged, the loop keeps the previous judgement
            logger.warning("judge query at t=%.1f failed: %s", t, e)
            return
        self.progress_max = max(self.progress_max, judgement.progress)
        self.latest = judgement
        self.history.append(dict(t=round(t, 3), **judgement.to_dict(), progress_max=self.progress_max, **meta))

    def reset(self) -> None:
        for buf in self.buffers.values():
            buf.clear()
        self.last_kept_t.clear()
        self.latest, self.history, self.progress_max = None, [], 0.0

    def close(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=self.cfg.timeout_s)
