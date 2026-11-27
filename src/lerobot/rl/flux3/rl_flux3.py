"""FLUX 3 Action in the fork's offline loop: ``Flux3Config`` plus the fields ``rl_offline`` reads (``flux3_rl``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import PreTrainedConfig
from lerobot.policies.flux3 import Flux3Config
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rl.shared_config import MemoryConfig

SPEED_LEVELS = ("slow", "normal", "fast")


@PreTrainedConfig.register_subclass("flux3_rl")
@dataclass
class Flux3RLConfig(Flux3Config):
    task: str = ""
    push_to_hub: bool = False
    offline_steps: int = 10000
    gradient_accumulation_steps: int = 1
    # The buffer's lookback supplies the older image snapshots: history_keys = the cameras,
    # history_offsets_seconds = the observation times before now.
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage_device: str = "cpu"
    offline_buffer_capacity: int = 1_000_000
    image_storage_dtype: str = "uint8"
    image_storage_size: tuple[int, int] = (448, 448)
    image_stride: int = 1
    reward_normalization_constant: float = 1.0
    terminal_failure_reward: float = -1.0
    async_prefetch: bool = True
    scheduler_groups: list[str] = field(default_factory=lambda: ["policy"])
    # The trunk weights are bf16: a plain AdamW write rounds most updates away.
    optimizer_stochastic_rounding: bool = True
    # -- run time (rl/rtc_actor_runtime.py) --
    # The chunk queue's settings; the sampler itself takes no correction from the last chunk yet.
    rtc_config: RTCConfig = field(default_factory=RTCConfig)
    # The caption's speed clause, one of SPEED_LEVELS.
    eval_speed: str = "normal"
    # The rollout's steps in order, as the subtask console and the sim scorer walk them. The caption
    # uses the text only; precision and contact are accepted because the sim tasks pass them.
    eval_subtasks: list[str] = field(default_factory=list)
    eval_home_subtask: str = "return to home"
    eval_subtask_precisions: list[int] | None = None
    eval_subtask_contacts: list[str] | None = None
    eval_home_precision: int = 1
    eval_home_contact: str = "na"


def render_caption(task: str, subtask: str, mistake: bool, speed: str) -> str:
    mistakes = "Mistakes are made." if mistake else "No mistakes are made."
    return f"The task is to {task}. The current step is {subtask}. {mistakes} The speed is {speed}."
