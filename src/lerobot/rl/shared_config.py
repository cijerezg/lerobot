from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Short-term memory (observation history) settings, model-agnostic.

    history_keys: observation keys to build lookback windows for (empty = disabled);
        the key "action" adds the executed actions at the same past frames, and
        "depth.{cam}.depth" (the buffer's canonical depth key) adds raw depth frames.
    history_window_seconds: how far back the window reaches.
    history_num_samples: number of past samples, evenly spaced over the window.
    """

    history_keys: list[str] = field(default_factory=list)
    history_window_seconds: float = 4.0
    history_num_samples: int = 4
    # Long-term memory: max completed subtasks kept in the per-frame done-list
    # (complementary_info["done_list_ids"], derived from subtask_index).
    done_list_cap: int = 10
    # π0.7-style metadata steering: label offline episodes at buffer fill
    # (quality/mistake defaults for curated demos, speed = length bucket) and
    # prompt quality=5 / mistake=false at inference.
    metadata_enabled: bool = False
    metadata_default_quality: int = 5
    metadata_default_mistake: bool = False
    metadata_speed_bucket_steps: int = 500

    def history_offsets(self, fps: float) -> dict[str, list[int]] | None:
        """Per-key lookback distances in buffer steps, e.g. 4 s / 4 samples @ 30 fps → [30, 60, 90, 120]."""
        if not self.history_keys:
            return None
        stride = self.history_window_seconds * fps / self.history_num_samples
        offsets = [round(stride * i) for i in range(1, self.history_num_samples + 1)]
        return dict.fromkeys(self.history_keys, offsets)


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 120
    queue_get_timeout: float = 2


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"
