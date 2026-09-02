"""Rate-aware future gripper-event targets shared by dataset converters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

MIN_CLOSED_DURATION_S = 0.5
TARGET_HALF_LIFE_S = 1.0
TARGET_CUTOFF_S = 5.0


def frames_for_duration(seconds: float, fps: float) -> int:
    """Convert a locked duration to an integral number of frames."""
    if seconds < 0 or fps <= 0 or not np.isfinite(seconds) or not np.isfinite(fps):
        raise ValueError(f"Invalid duration {seconds!r} or FPS {fps!r}")
    frames = seconds * fps
    rounded = int(round(frames))
    if not np.isclose(frames, rounded, atol=1e-9):
        raise ValueError(f"Duration {seconds:g}s is not an integral frame count at {fps:g} FPS")
    return rounded


def retained_closed_intervals(
    closed_mask: np.ndarray, *, min_closed_frames: int
) -> list[tuple[int, int]]:
    """Return half-open true runs that satisfy the persistence threshold."""
    mask = np.asarray(closed_mask)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("closed_mask must be a one-dimensional boolean array")
    if min_closed_frames <= 0:
        raise ValueError("min_closed_frames must be positive")

    padded = np.concatenate(([False], mask, [False]))
    transitions = np.flatnonzero(padded[1:] != padded[:-1])
    intervals = [
        (int(start), int(stop))
        for start, stop in transitions.reshape(-1, 2)
        if stop - start >= min_closed_frames
    ]
    return intervals


def future_event_targets(
    length: int,
    event_frames: Sequence[int],
    *,
    half_life_frames: int,
    cutoff_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cutoff next-event deltas and exponential targets."""
    if length < 0 or half_life_frames <= 0 or not 0 <= cutoff_frames <= np.iinfo(np.int16).max:
        raise ValueError("Invalid target length, half-life, or cutoff")
    event_mask = np.zeros(length, dtype=bool)
    event_array = np.asarray(event_frames, dtype=np.int64)
    if event_array.size:
        if (
            event_array.min() < 0
            or event_array.max() >= length
            or len(np.unique(event_array)) != len(event_array)
        ):
            raise ValueError("Event frames must be unique and inside the episode")
        event_mask[event_array] = True

    deltas = np.full(length, -1, dtype=np.int16)
    next_event = -1
    for frame in range(length - 1, -1, -1):
        if event_mask[frame]:
            next_event = frame
        if next_event >= 0 and next_event - frame <= cutoff_frames:
            deltas[frame] = next_event - frame

    targets = np.zeros(length, dtype=np.float32)
    valid = deltas >= 0
    targets[valid] = np.exp2(-deltas[valid].astype(np.float32) / half_life_frames).astype(
        np.float32
    )
    return deltas, targets


def depth_gripper_event_labels_from_closed_mask(
    closed_mask: np.ndarray,
    *,
    fps: float,
) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    """Build REBOT-compatible close/open targets from a dataset-native closed mask."""
    min_closed_frames = frames_for_duration(MIN_CLOSED_DURATION_S, fps)
    half_life_frames = frames_for_duration(TARGET_HALF_LIFE_S, fps)
    cutoff_frames = frames_for_duration(TARGET_CUTOFF_S, fps)
    intervals = retained_closed_intervals(
        np.asarray(closed_mask), min_closed_frames=min_closed_frames
    )
    length = len(closed_mask)
    close_frames = [start for start, _ in intervals if start > 0]
    open_frames = [stop for _, stop in intervals if stop < length]
    close_delta, close_target = future_event_targets(
        length,
        close_frames,
        half_life_frames=half_life_frames,
        cutoff_frames=cutoff_frames,
    )
    open_delta, open_target = future_event_targets(
        length,
        open_frames,
        half_life_frames=half_life_frames,
        cutoff_frames=cutoff_frames,
    )
    return (
        {
            "depth_gripper_close_target": close_target,
            "depth_gripper_open_target": open_target,
            "depth_gripper_close_delta": close_delta,
            "depth_gripper_open_delta": open_delta,
        },
        intervals,
    )
