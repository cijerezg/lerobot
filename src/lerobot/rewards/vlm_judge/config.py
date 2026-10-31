from dataclasses import dataclass, field


@dataclass
class JudgeConfig:
    """Where the judge runs and what one query carries. Model-agnostic: anything that serves
    /v1/chat/completions with image_url parts (vllm, or lerobot/scripts/spark_judge_hf_server.py)."""

    endpoint: str = "http://localhost:8011/v1"  # sims, scoring and models all run on the Spark (user 2026-10-30)
    model: str = "google/gemma-4-31B-it"
    cameras: list[str] = field(default_factory=lambda: ["top", "wrist"])
    frames_per_camera: int = 4  # the strip = the last N frames of each camera, oldest first -> 8 images by default
    frame_period_s: float = 0.5  # spacing of the frames in the strip
    image_max_side: int = 640  # frames are resized so the long side is at most this (640x480 inputs pass through)
    jpeg_quality: int = 85
    every_k_steps: int = 15  # runner: issue a query every K control steps (30 Hz -> 0.5 s)
    timeout_s: float = 60.0
    max_tokens: int = 300
    temperature: float = 0.0
    # sent as the request's chat_template_kwargs (vllm): Qwen3.5 thinks by default and Gemma 4 can; both answer directly with this
    chat_template_kwargs: dict = field(default_factory=lambda: {"enable_thinking": False})


ASSIGNER = dict(endpoint="http://localhost:8013/v1", model="Qwen/Qwen3.5-9B", frames_per_camera=3, frame_period_s=0.5, max_tokens=40)
