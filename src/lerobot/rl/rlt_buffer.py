from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RLTReplaySample:
    rl_token: Tensor
    proprio: Tensor
    reference_chunk: Tensor
    executed_chunk: Tensor
    next_rl_token: Tensor
    next_proprio: Tensor
    next_reference_chunk: Tensor
    reward: float
    done: bool
    is_intervention: bool


class RLTReplayBuffer:
    """Small tensor replay for online RLT head training."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._samples: deque[RLTReplaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: RLTReplaySample) -> None:
        self._samples.append(
            RLTReplaySample(
                rl_token=sample.rl_token.detach().cpu(),
                proprio=sample.proprio.detach().cpu(),
                reference_chunk=sample.reference_chunk.detach().cpu(),
                executed_chunk=sample.executed_chunk.detach().cpu(),
                next_rl_token=sample.next_rl_token.detach().cpu(),
                next_proprio=sample.next_proprio.detach().cpu(),
                next_reference_chunk=sample.next_reference_chunk.detach().cpu(),
                reward=float(sample.reward),
                done=bool(sample.done),
                is_intervention=bool(sample.is_intervention),
            )
        )

    def sample(self, batch_size: int, *, device: torch.device | str | None = None) -> dict[str, Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if len(self._samples) < batch_size:
            raise ValueError(f"not enough samples: {len(self._samples)} < {batch_size}")

        samples = random.sample(list(self._samples), batch_size)
        out = {
            "rl_token": torch.stack([s.rl_token for s in samples], dim=0),
            "proprio": torch.stack([s.proprio for s in samples], dim=0),
            "reference_chunk": torch.stack([s.reference_chunk for s in samples], dim=0),
            "executed_chunk": torch.stack([s.executed_chunk for s in samples], dim=0),
            "next_rl_token": torch.stack([s.next_rl_token for s in samples], dim=0),
            "next_proprio": torch.stack([s.next_proprio for s in samples], dim=0),
            "next_reference_chunk": torch.stack([s.next_reference_chunk for s in samples], dim=0),
            "reward": torch.tensor([s.reward for s in samples], dtype=torch.float32).unsqueeze(-1),
            "done": torch.tensor([s.done for s in samples], dtype=torch.float32).unsqueeze(-1),
            "is_intervention": torch.tensor(
                [s.is_intervention for s in samples], dtype=torch.float32
            ).view(-1, 1, 1),
        }
        if device is not None:
            out = {key: value.to(device) for key, value in out.items()}
        return out
