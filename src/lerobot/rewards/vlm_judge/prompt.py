"""Builds the OpenAI chat-completions request: one system message with the task, the vocabulary and the ladder, one
user message interleaving "camera at t" labels and the frames (oldest first), and the JSON answer contract."""

import base64
import io
import json
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .task_spec import TaskSpec

ANSWER_KEYS = ("subtask_id", "stage_id", "progress_0_1", "confidence", "evidence")


@dataclass
class Frame:
    t: float  # seconds since the episode start
    camera: str  # "top" | "wrist"
    image: np.ndarray  # HxWx3 uint8 RGB


def encode_jpeg(image: np.ndarray, max_side: int, quality: int) -> str:
    img = Image.fromarray(image)
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def system_prompt(spec: TaskSpec) -> str:
    subtasks = "\n".join(
        f"  {s.id}: {s.text} -- contact {s.contact}, precision {s.precision}/5. Visible when: {s.criterion}" for s in spec.subtasks
    )
    stages = "\n".join(f"  {s.id}: {s.name} (value {s.value:.1f}) -- {s.criterion}" for s in spec.stages)
    return (
        "You judge short strips of frames from a single-arm tabletop robot (7-DOF arm, parallel two-finger gripper) doing one task. "
        "Two cameras: 'top' looks down over the table from the far edge, 'wrist' is mounted on the gripper and looks at the fingers. "
        "Frames are given oldest first with their time in seconds. Judge the LATEST frame; use the earlier frames only to read motion "
        "direction and what has already happened.\n\n"
        f"Task: {spec.task}\n\n"
        "Subtask vocabulary (pick exactly one id, the one whose criterion the latest frame satisfies):\n"
        f"{subtasks}\n\n"
        "Stage ladder (cumulative; pick the HIGHEST stage id whose criterion the latest frame satisfies; stage 0 if none):\n"
        f"{stages}\n\n"
        "progress_0_1 is a number between the value of the chosen stage and the value of the next stage, placing the latest frame "
        "between them (exactly the stage value when the stage was just reached). confidence is your probability (0-1) that the "
        "subtask_id is right. evidence is one sentence naming what you see in the latest frame that decides the answer. "
        "Do not guess beyond what is visible: when the fingers or the object are hidden, prefer the lower stage and say so.\n\n"
        'Answer with one JSON object and nothing else: {"subtask_id": int, "stage_id": int, "progress_0_1": float, '
        '"confidence": float, "evidence": str}'
    )


def build_messages(spec: TaskSpec, frames: list[Frame], max_side: int = 640, quality: int = 85) -> list[dict]:
    frames = sorted(frames, key=lambda f: (f.t, f.camera))
    content = [{"type": "text", "text": f"Strip of {len(frames)} frames, latest at t={frames[-1].t:.1f} s."}]
    for f in frames:
        content.append({"type": "text", "text": f"{f.camera} camera, t={f.t:.1f} s:"})
        content.append({"type": "image_url", "image_url": {"url": encode_jpeg(f.image, max_side, quality)}})
    content.append({"type": "text", "text": "Return the JSON object."})
    return [{"role": "system", "content": system_prompt(spec)}, {"role": "user", "content": content}]


def build_request(spec: TaskSpec, frames: list[Frame], model: str, max_tokens: int, temperature: float,
                  max_side: int = 640, quality: int = 85, chat_template_kwargs: dict | None = None) -> dict:
    return dict(
        model=model,
        messages=build_messages(spec, frames, max_side, quality),
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
        chat_template_kwargs=chat_template_kwargs or {},
    )


def describe_request(request: dict) -> str:
    """The request with the image payloads replaced by their byte counts, for logs and tests."""
    out = json.loads(json.dumps(request))
    for part in out["messages"][1]["content"]:
        if part["type"] == "image_url":
            part["image_url"] = f"<{len(part['image_url']['url'])} chars>"
    return json.dumps(out, indent=1)
