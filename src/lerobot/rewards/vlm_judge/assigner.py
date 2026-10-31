"""Live subtask assigner: a small VLM (Qwen3.5-9B on the Spark, ``spark_judge_serve.sh assigner``) names the step of the
policy's script the robot is working on, from the last few frames. Its answer is the script cursor the policy is
conditioned on (``eval_subtasks``: tasks.py in sim, config_rl.yaml on hardware), so the vocabulary IS the script.

Only the steps that can be current are offered (user 2026-10-30: no more subtasks than needed). The script splits into
segments, one per object: a new segment starts at every "grasp ..." step (``segments``). With the cursor in segment k,
the model sees segment k's steps plus the first step of segment k + 1, so it can hold, go back inside the segment (a
dropped object -> grasp again) or move on; the step finished just before the segment is named as context. A 3-step
basket task is one segment (3-4 options); the 20-step hardware script never offers more than 5.

Per step the model gets one "working on it while" line built from the verb (``step_criterion``); the same lines are the
subtask criteria of the Gemma specs (sim_spec.py). Answer: ``{"step": <offered id>, "confidence": <0-1>}`` (plus
``evidence`` when ``with_evidence``), ~15 tokens."""

import logging
import re
import threading
import time
from dataclasses import asdict, dataclass

import requests

from .config import JudgeConfig
from .parse import JudgeParseError, _extract_json
from .prompt import Frame, encode_jpeg
from .runner import JudgeRunner

logger = logging.getLogger(__name__)

_STEP = re.compile(r"^(?P<verb>\w+)\s+(?P<obj>.+?)(?:\s+(?P<prep>to|into|in|over)\s+(?P<target>.+))?$")

# verb -> what the frames show while the robot is working on that step ({obj}, {target} from the step text)
WHILE = {
    "grasp": "the gripper moves toward {obj} or closes on it; {obj} still rests where it was",
    "move": "{obj} is held between the fingers, off the table, travelling toward {target}",
    "release": "{obj} is held at or above {target}; the fingers open and let it go",
    "push": "{obj} sits in {target} and the fingers press it down to seat it",
    "close": "the fingers hold {obj} and swing it shut",
    "fold": "{obj} is held and swings over {target}",
    "press": "everything is folded and the gripper presses down on the closed kit to flatten it",
    "return": "the steps above are finished, the gripper is empty and the arm moves back to its rest pose",
}


def step_criterion(step: str) -> str:
    m = _STEP.match(step.strip())
    if m is None or m["verb"] not in WHILE:
        return step
    target = m["target"] or "its goal"
    return WHILE[m["verb"]].format(obj=m["obj"], target=target)


def segments(script: list[str]) -> list[tuple[int, int]]:
    """[start, end) per object: a segment starts at step 0 and at every 'grasp ...' step."""
    starts = [0] + [i for i, s in enumerate(script) if i > 0 and s.startswith("grasp ")]
    return list(zip(starts, starts[1:] + [len(script)], strict=True))


def offered_steps(script: list[str], cursor: int) -> list[int]:
    """Script indices the assigner may answer with when the current cursor is ``cursor``."""
    segs = segments(script)
    k = next(i for i, (a, b) in enumerate(segs) if a <= cursor < b)
    a, b = segs[k]
    return list(range(a, b)) + ([segs[k + 1][0]] if k + 1 < len(segs) else [])


@dataclass
class Assignment:
    cursor: int  # script index
    step: str
    confidence: float
    offered: list[int]
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def system_prompt(task: str, script: list[str], offered: list[int], with_evidence: bool) -> str:
    lines = "\n".join(f"  {k}: {script[i]} -- working on it while {step_criterion(script[i])}" for k, i in enumerate(offered))
    before = f"Already finished before these steps: {script[offered[0] - 1]}.\n" if offered[0] > 0 else ""
    answer = '{"step": int, "confidence": float, "evidence": str}' if with_evidence else '{"step": int, "confidence": float}'
    return (
        "You watch a single-arm tabletop robot (parallel two-finger gripper) through two cameras: 'top' looks down over the table "
        "from the far edge, 'wrist' is on the gripper and looks at the fingers. Frames come oldest first with their time in seconds.\n\n"
        f"Task: {task}\n{before}"
        "Steps, in the order the robot does them:\n"
        f"{lines}\n\n"
        "Pick the step the robot is working on in the LATEST frame; use the earlier frames to see the motion. A step is finished only "
        "when what it describes is visibly done; when unsure, keep the earlier step. confidence = your probability that the step is right."
        + (" evidence = at most 12 words on what decides it." if with_evidence else "")
        + f"\n\nAnswer with one JSON object and nothing else: {answer}"
    )


def build_request(cfg: JudgeConfig, task: str, script: list[str], offered: list[int], frames: list[Frame], with_evidence: bool) -> dict:
    frames = sorted(frames, key=lambda f: (f.t, f.camera))
    content = []
    for f in frames:
        content.append({"type": "text", "text": f"{f.camera} camera, t={f.t:.1f} s:"})
        content.append({"type": "image_url", "image_url": {"url": encode_jpeg(f.image, cfg.image_max_side, cfg.jpeg_quality)}})
    content.append({"type": "text", "text": "Return the JSON object."})
    return dict(
        model=cfg.model,
        messages=[{"role": "system", "content": system_prompt(task, script, offered, with_evidence)}, {"role": "user", "content": content}],
        max_tokens=cfg.max_tokens, temperature=cfg.temperature, response_format={"type": "json_object"},
        chat_template_kwargs=cfg.chat_template_kwargs,
    )


def parse_assignment(text: str, script: list[str], offered: list[int]) -> Assignment:
    raw = _extract_json(text)
    if not isinstance(raw, dict) or "step" not in raw or "confidence" not in raw:
        raise JudgeParseError(f"need keys step, confidence: {text[:200]!r}")
    k, conf = raw["step"], raw["confidence"]
    if isinstance(k, bool) or not isinstance(k, (int, float)) or float(k) != int(k) or not 0 <= int(k) < len(offered):
        raise JudgeParseError(f"step must be an integer in 0..{len(offered) - 1}, got {k!r}")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        raise JudgeParseError(f"confidence must be in [0, 1], got {conf!r}")
    cursor = offered[int(k)]
    return Assignment(cursor=cursor, step=script[cursor], confidence=float(conf), offered=list(offered), evidence=str(raw.get("evidence", "")).strip())


class AssignerClient:
    def __init__(self, cfg: JudgeConfig, task: str, script: list[str], with_evidence: bool = False) -> None:
        self.cfg, self.task, self.script, self.with_evidence = cfg, task, list(script), with_evidence
        self.session = requests.Session()

    def query(self, cursor: int, frames: list[Frame]) -> tuple[Assignment, dict]:
        offered = offered_steps(self.script, cursor)
        request = build_request(self.cfg, self.task, self.script, offered, frames, self.with_evidence)
        t0 = time.perf_counter()
        r = self.session.post(f"{self.cfg.endpoint.rstrip('/')}/chat/completions", json=request, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        return parse_assignment(text, self.script, offered), dict(latency_s=round(time.perf_counter() - t0, 3), raw=text, usage=body.get("usage"))


class AssignerRunner(JudgeRunner):
    """JudgeRunner's frame buffer and one-query-in-flight worker; ``cursor`` is the cursor the runtime currently holds (set it
    every step), and ``latest.cursor`` is the assigner's newest answer for the runtime to seek to."""

    def __init__(self, cfg: JudgeConfig, client: AssignerClient) -> None:
        super().__init__(cfg, spec=None, client=client)
        self.cursor = 0

    def maybe_query(self, t: float) -> bool:
        if self.thread is not None and self.thread.is_alive():
            return False
        frames = [f for buf in self.buffers.values() for f in buf]
        if not frames:
            return False
        self.thread = threading.Thread(target=self._query, args=(t, frames, self.cursor), daemon=True)
        self.thread.start()
        return True

    def _query(self, t: float, frames: list[Frame], cursor: int) -> None:
        try:
            assignment, meta = self.client.query(cursor, frames)
        except Exception as e:
            logger.warning("assigner query at t=%.1f failed: %s", t, e)
            return
        self.latest = assignment
        self.history.append(dict(t=round(t, 3), from_cursor=cursor, **assignment.to_dict(), **meta))

    def reset(self) -> None:
        super().reset()
        self.cursor = 0
