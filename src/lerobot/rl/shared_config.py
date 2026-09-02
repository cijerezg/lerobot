from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryConfig:
    """Short-term memory (observation history) settings, model-agnostic.

    history_keys: observation keys to build lookback windows for (empty = disabled);
        the key "action" adds the executed actions at the same past frames, and
        "depth.{cam}.depth" (the buffer's canonical depth key) adds raw depth frames.
    history_offsets_seconds: the lookback instants themselves, in seconds before now.
        This is the source of truth for the window shape. Written the way the offsets
        are talked about and the way the diverse corpus stores them ([-6, -4, -2]);
        a positive list ([6, 4, 2]) means the same thing. Order does not matter — the
        list is normalized to magnitudes sorted oldest → newest, matching the slot
        order every consumer emits.
    history_window_seconds / history_num_samples: the uniform parameterization, and
        the default when history_offsets_seconds is None — an evenly spaced window,
        which is also how every checkpoint written before the explicit list loads. When
        history_offsets_seconds IS set they are recomputed from it (span and count) so
        downstream code reading either field stays consistent. The defaults below spell
        the same window as the explicit [-6, -4, -2]: 6 s / 3 samples → 2 s stride.
    """

    history_keys: list[str] = field(default_factory=list)
    history_offsets_seconds: list[float] | None = None
    history_window_seconds: float = 6.0
    history_num_samples: int = 3
    # Optional dropout on the whole consumption-side history block (training only).
    # Default off so the train-time observation matches deployment.
    history_dropout: float = 0.0
    # π0.7-style metadata steering: per-episode quality + per-window mistake
    # loaded from the dataset (metadata_annotate.py; speed omitted) and
    # prompt quality=5 / mistake=false at inference.
    metadata_enabled: bool = False

    def __post_init__(self) -> None:
        if self.history_num_samples < 0:
            raise ValueError(f"history_num_samples must be >= 0, got {self.history_num_samples}.")
        if self.history_keys and self.history_num_samples == 0:
            raise ValueError("history_num_samples must be >= 1 when history_keys are enabled.")
        if self.history_window_seconds <= 0:
            raise ValueError(f"history_window_seconds must be > 0, got {self.history_window_seconds}.")
        if not 0 <= self.history_dropout < 1:
            raise ValueError(f"history_dropout must be in [0, 1), got {self.history_dropout}.")
        if self.history_offsets_seconds is not None:
            self.history_offsets_seconds = self._normalize_offsets_seconds(self.history_offsets_seconds)
            # The explicit list wins: keep the legacy pair describing the same window so
            # every reader of history_window_seconds (the probes' "stale" lag) or
            # history_num_samples (the prompt token budget) agrees with the slots.
            self.history_window_seconds = self.history_offsets_seconds[0]
            self.history_num_samples = len(self.history_offsets_seconds)

    @staticmethod
    def _normalize_offsets_seconds(offsets: list[float]) -> list[float]:
        """Sign-agnostic magnitudes, sorted oldest → newest, deduplicated."""
        values = [float(o) for o in offsets]
        if not values:
            raise ValueError("history_offsets_seconds must not be empty; use None for the uniform window.")
        if any(o == 0.0 for o in values):
            raise ValueError(f"history_offsets_seconds cannot contain 0 (the current frame), got {values}.")
        if min(values) < 0 < max(values):
            raise ValueError(
                f"history_offsets_seconds must be all-negative (seconds before now) or "
                f"all-positive (lookback magnitudes), not mixed, got {values}."
            )
        magnitudes = sorted({abs(o) for o in values}, reverse=True)
        if len(magnitudes) != len(values):
            raise ValueError(f"history_offsets_seconds must be distinct, got {values}.")
        return magnitudes

    def history_times_seconds(self) -> list[float]:
        """Slot ages in seconds before now, oldest → newest — the real-valued instants
        the sinusoidal e(t) stamps are built from (MEM video encoder, depth
        TemporalFusion). E.g. [6.0, 4.0, 2.0]."""
        if self.history_offsets_seconds is not None:
            return list(self.history_offsets_seconds)
        if self.history_num_samples <= 0:
            return []  # history disabled; no slots to stamp
        stride = self.history_window_seconds / self.history_num_samples
        return [stride * (self.history_num_samples - i) for i in range(self.history_num_samples)]

    def history_offsets_frames(self, fps: float) -> list[int]:
        """history_times_seconds converted to lookback distances in buffer steps,
        oldest → newest, e.g. 6/4/2 s @ 30 fps → [180, 120, 60]."""
        offsets = [round(t * fps) for t in self.history_times_seconds()]
        if not offsets:
            return []
        if len(set(offsets)) != len(offsets) or min(offsets) <= 0:
            raise ValueError(
                f"history offsets {self.history_times_seconds()} s collapse to {offsets} "
                f"steps at {fps} fps — slots must be distinct and at least one frame back."
            )
        return offsets

    def history_offsets(self, fps: float) -> dict[str, list[int]] | None:
        """Per-key lookback distances in buffer steps, e.g. -6/-4/-2 s @ 30 fps →
        [180, 120, 60] (oldest → newest, as the buffer normalizes them)."""
        if not self.history_keys:
            return None
        return dict.fromkeys(self.history_keys, self.history_offsets_frames(fps))


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
