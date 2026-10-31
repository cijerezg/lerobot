"""Robometer-4B (lerobot/Robometer-4B, Qwen3-VL-4B + progress/success heads) as a second progress signal. The model
runs on the Spark in its own venv behind ``server.py`` (port 8012, upstream lerobot's inference path); this package
is the HTTP client and the episode CLI. Read migration/digital_twin_2026-10-29/judge/JUDGE_REPORT.md first."""

from .client import RobometerClient as RobometerClient
