from __future__ import annotations

import json
import logging
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    # v2 review-only fields. None on v1 buffers and when capture is disabled,
    # so the training hot path never has to branch on them.
    images_jpeg: dict[str, bytes] | None = None
    inference_ts: float | None = None
    episode_id: int | None = None
    success: bool | None = None
    failure: bool | None = None
    chunk_start_step: int | None = None


# Bumped whenever new fields are persisted. Loaders must remain backward
# compatible with all prior versions by defaulting missing keys to None.
RLT_REPLAY_BUFFER_VERSION = 2
RLT_REVIEW_SIDECAR_VERSION = 1
_RLT_REVIEW_LABELS = {"success", "failure", "open"}
_LOGGER = logging.getLogger(__name__)


def _copy_images_jpeg(images: dict[str, bytes] | None) -> dict[str, bytes] | None:
    if images is None:
        return None
    # JPEG payloads are immutable bytes; the dict is the only mutable layer.
    return dict(images)


def _review_sidecar_path(replay_path: Path) -> Path:
    return replay_path.with_suffix(".review.json")


def _warn_malformed_review_entry(sidecar_path: Path, episode_id: object, reason: str) -> None:
    _LOGGER.warning(
        "Ignoring malformed RLT review sidecar entry in %s for episode %r: %s",
        sidecar_path,
        episode_id,
        reason,
    )


def _load_review_sidecar_entries(
    sidecar_path: Path, valid_episode_ids: set[int]
) -> dict[int, tuple[str, bool]]:
    try:
        with sidecar_path.open("r", encoding="utf-8") as f:
            sidecar = json.load(f)
    except OSError as exc:
        _LOGGER.warning("Failed to read RLT review sidecar %s: %s", sidecar_path, exc)
        return {}
    except json.JSONDecodeError as exc:
        _LOGGER.warning("Ignoring malformed RLT review sidecar %s: %s", sidecar_path, exc)
        return {}

    if not isinstance(sidecar, dict):
        _LOGGER.warning("Ignoring malformed RLT review sidecar %s: expected object", sidecar_path)
        return {}
    if sidecar.get("version") != RLT_REVIEW_SIDECAR_VERSION:
        _LOGGER.warning(
            "Ignoring unsupported RLT review sidecar %s version %r",
            sidecar_path,
            sidecar.get("version"),
        )
        return {}

    episodes = sidecar.get("episodes")
    if not isinstance(episodes, dict):
        _LOGGER.warning("Ignoring malformed RLT review sidecar %s: expected episodes object", sidecar_path)
        return {}

    entries: dict[int, tuple[str, bool]] = {}
    for episode_key, entry in episodes.items():
        try:
            episode_id = int(episode_key)
        except (TypeError, ValueError):
            _warn_malformed_review_entry(sidecar_path, episode_key, "episode id must be an integer")
            continue
        if episode_id not in valid_episode_ids:
            continue
        if not isinstance(entry, dict):
            _warn_malformed_review_entry(sidecar_path, episode_key, "entry must be an object")
            continue
        if "label" not in entry or "deleted" not in entry:
            _warn_malformed_review_entry(sidecar_path, episode_key, "entry must include label and deleted")
            continue

        label = entry["label"]
        deleted = entry["deleted"]
        if not isinstance(label, str) or label not in _RLT_REVIEW_LABELS:
            _warn_malformed_review_entry(
                sidecar_path,
                episode_key,
                f"label must be one of {sorted(_RLT_REVIEW_LABELS)}",
            )
            continue
        if not isinstance(deleted, bool):
            _warn_malformed_review_entry(sidecar_path, episode_key, "deleted must be a boolean")
            continue
        entries[episode_id] = (label, deleted)
    return entries


class RLTReplayBuffer:
    """Small tensor replay for online RLT head training."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._samples: deque[RLTReplaySample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def capacity(self) -> int:
        return int(self._samples.maxlen or 0)

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
                images_jpeg=_copy_images_jpeg(sample.images_jpeg),
                inference_ts=None if sample.inference_ts is None else float(sample.inference_ts),
                episode_id=None if sample.episode_id is None else int(sample.episode_id),
                success=None if sample.success is None else bool(sample.success),
                failure=None if sample.failure is None else bool(sample.failure),
                chunk_start_step=None
                if sample.chunk_start_step is None
                else int(sample.chunk_start_step),
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

    def samples(self) -> list[RLTReplaySample]:
        """Return a CPU snapshot of the stored samples in replay order."""
        return list(self._samples)

    def extend(self, samples: list[RLTReplaySample]) -> None:
        for sample in samples:
            self.add(sample)

    def state_dict(self) -> dict[str, Any]:
        def _sample_state(sample: RLTReplaySample) -> dict[str, Any]:
            state: dict[str, Any] = {
                "rl_token": sample.rl_token.detach().cpu(),
                "proprio": sample.proprio.detach().cpu(),
                "reference_chunk": sample.reference_chunk.detach().cpu(),
                "executed_chunk": sample.executed_chunk.detach().cpu(),
                "next_rl_token": sample.next_rl_token.detach().cpu(),
                "next_proprio": sample.next_proprio.detach().cpu(),
                "next_reference_chunk": sample.next_reference_chunk.detach().cpu(),
                "reward": float(sample.reward),
                "done": bool(sample.done),
                "is_intervention": bool(sample.is_intervention),
                "images_jpeg": _copy_images_jpeg(sample.images_jpeg),
                "inference_ts": None if sample.inference_ts is None else float(sample.inference_ts),
                "episode_id": None if sample.episode_id is None else int(sample.episode_id),
                "success": None if sample.success is None else bool(sample.success),
                "failure": None if sample.failure is None else bool(sample.failure),
                "chunk_start_step": None
                if sample.chunk_start_step is None
                else int(sample.chunk_start_step),
            }
            return state

        return {
            "version": RLT_REPLAY_BUFFER_VERSION,
            "capacity": self.capacity,
            "samples": [_sample_state(sample) for sample in self._samples],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        samples = state_dict.get("samples", [])
        self._samples.clear()
        for sample in samples:
            self.add(
                RLTReplaySample(
                    rl_token=sample["rl_token"],
                    proprio=sample["proprio"],
                    reference_chunk=sample["reference_chunk"],
                    executed_chunk=sample["executed_chunk"],
                    next_rl_token=sample["next_rl_token"],
                    next_proprio=sample["next_proprio"],
                    next_reference_chunk=sample["next_reference_chunk"],
                    reward=float(sample["reward"]),
                    done=bool(sample["done"]),
                    is_intervention=bool(sample["is_intervention"]),
                    images_jpeg=sample.get("images_jpeg"),
                    inference_ts=sample.get("inference_ts"),
                    episode_id=sample.get("episode_id"),
                    success=sample.get("success"),
                    failure=sample.get("failure"),
                    chunk_start_step=sample.get("chunk_start_step"),
                )
            )

    def apply_review_sidecar(self, replay_path: str | Path) -> None:
        sidecar_path = _review_sidecar_path(Path(replay_path))
        if not sidecar_path.exists():
            return

        valid_episode_ids = {int(sample.episode_id) for sample in self._samples if sample.episode_id is not None}
        review_entries = _load_review_sidecar_entries(sidecar_path, valid_episode_ids)
        if not review_entries:
            return

        before_count = len(self._samples)
        samples = [
            sample
            for sample in self._samples
            if sample.episode_id is None or not review_entries.get(int(sample.episode_id), ("open", False))[1]
        ]
        samples_by_episode: dict[int, list[int]] = {}
        for index, sample in enumerate(samples):
            if sample.episode_id is None:
                continue
            episode_id = int(sample.episode_id)
            entry = review_entries.get(episode_id)
            if entry is None or entry[1]:
                continue
            samples_by_episode.setdefault(episode_id, []).append(index)

        for episode_id, indices in samples_by_episode.items():
            label = review_entries[episode_id][0]
            terminal_index = next(
                (index for index in reversed(indices) if samples[index].done),
                indices[-1],
            )
            for index in indices:
                sample = samples[index]
                sample.success = label == "success"
                sample.failure = label == "failure"
                if label == "open":
                    sample.done = False
                    sample.reward = 0.0
                elif index == terminal_index:
                    sample.done = True
                    sample.reward = 1.0 if label == "success" else 0.0
                else:
                    sample.done = False
                    sample.reward = 0.0

        self._samples = deque(samples, maxlen=self.capacity)
        _LOGGER.info(
            "Applied RLT review sidecar %s: samples %d -> %d, updated_episodes=%d",
            sidecar_path,
            before_count,
            len(self._samples),
            len(review_entries),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        capacity: int | None = None,
        apply_review_sidecar: bool = False,
    ) -> RLTReplayBuffer:
        # `weights_only=True` is safe here because every persisted value is a
        # tensor, plain Python scalar, str, bytes, or dict/list of those types.
        replay_path = Path(path)
        state_dict = torch.load(replay_path, map_location="cpu", weights_only=True)
        replay_capacity = int(capacity or state_dict.get("capacity", 1))
        replay = cls(capacity=replay_capacity)
        replay.load_state_dict(state_dict)
        if apply_review_sidecar:
            replay.apply_review_sidecar(replay_path)
        return replay
