"""VLM judge: a local vision-language model behind an OpenAI-compatible HTTP endpoint assigns the current subtask
(fixed per-task vocabulary, tasks/*.json) and the stage on the hand-written scorer's ladder from a strip of recent
frames. Client only: the model runs on the Spark (lerobot/scripts/spark_judge_*.sh). Read
migration/digital_twin_2026-10-29/judge/JUDGE_REPORT.md first."""

from .client import JudgeClient as JudgeClient
from .config import JudgeConfig as JudgeConfig
from .parse import (
    Judgement as Judgement,
    JudgeParseError as JudgeParseError,
    parse_response as parse_response,
)
from .prompt import Frame as Frame, build_messages as build_messages
from .runner import JudgeRunner as JudgeRunner
from .task_spec import TaskSpec as TaskSpec
