from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActorLearnerConfig:
    learner_host: str = "127.0.0.1"
    learner_port: int = 50051
    policy_parameters_push_frequency: int = 4
    queue_get_timeout: float = 2


@dataclass
class ConcurrencyConfig:
    actor: str = "threads"
    learner: str = "threads"
