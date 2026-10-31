"""POST /score {task, frames: [jpeg data URIs oldest first]} -> {progress: [...], success: [...]} per frame."""

import time

import numpy as np
import requests

from lerobot.rewards.vlm_judge.prompt import encode_jpeg


class RobometerClient:
    def __init__(self, endpoint: str = "http://localhost:8012", timeout_s: float = 60.0, image_max_side: int = 640, jpeg_quality: int = 85) -> None:
        self.endpoint, self.timeout_s, self.image_max_side, self.jpeg_quality = endpoint.rstrip("/"), timeout_s, image_max_side, jpeg_quality
        self.session = requests.Session()

    def score(self, task: str, frames: list[np.ndarray]) -> tuple[dict, float]:
        """``frames`` = up to max_frames HxWx3 uint8 RGB of ONE camera, oldest first. Returns (body, latency_s)."""
        payload = dict(task=task, frames=[encode_jpeg(f, self.image_max_side, self.jpeg_quality) for f in frames])
        t0 = time.perf_counter()
        r = self.session.post(f"{self.endpoint}/score", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        assert len(body["progress"]) == len(frames) and len(body["success"]) == len(frames), body
        return body, round(time.perf_counter() - t0, 3)
