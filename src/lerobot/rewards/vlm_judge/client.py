"""One blocking HTTP query to the judge endpoint."""

import time

import requests

from .config import JudgeConfig
from .parse import Judgement, parse_response
from .prompt import Frame, build_request
from .task_spec import TaskSpec


class JudgeClient:
    def __init__(self, cfg: JudgeConfig) -> None:
        self.cfg = cfg
        self.session = requests.Session()

    def query(self, spec: TaskSpec, frames: list[Frame]) -> tuple[Judgement, dict]:
        """Returns the parsed judgement and a meta dict (latency_s, raw text, usage). Raises on HTTP or parse errors."""
        cfg = self.cfg
        request = build_request(spec, frames, cfg.model, cfg.max_tokens, cfg.temperature, cfg.image_max_side, cfg.jpeg_quality, cfg.chat_template_kwargs)
        t0 = time.perf_counter()
        r = self.session.post(f"{cfg.endpoint.rstrip('/')}/chat/completions", json=request, timeout=cfg.timeout_s)
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        judgement = parse_response(text, spec)
        return judgement, dict(latency_s=round(time.perf_counter() - t0, 3), raw=text, usage=body.get("usage"))
