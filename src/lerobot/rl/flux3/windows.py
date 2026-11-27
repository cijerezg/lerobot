"""Low-dimensional inputs of FLUX 3 Action from one rebot root, computed in float32 from the parquet files.

The replay buffer keeps states and actions in bf16, so the state and command windows are built here and
installed as buffer columns (``Flux3Trainer.prepare_buffer``). Everything is per dataset frame, in frame order.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.rl.flux3.rl_flux3 import SPEED_LEVELS
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.gripper_torque import smooth_gripper_torque
from lerobot.utils.motor_velocity import smooth_velocity

OBS_VELOCITY = "observation.velocity"
OBS_EFFORT = "observation.effort"


def read_frames(root) -> pd.DataFrame:
    files = sorted(Path(root).glob("data/*/*.parquet"))
    columns = ["index", "episode_index", ACTION, OBS_STATE, OBS_VELOCITY, OBS_EFFORT]
    return pd.concat(pd.read_parquet(f, columns=columns) for f in files).sort_values("index").reset_index(drop=True)


def episode_bounds(frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """First and last frame of each frame's episode."""
    episode = frames.episode_index.to_numpy()
    starts = np.flatnonzero(np.diff(episode, prepend=episode[0] - 1))
    ends = np.append(starts[1:], len(episode)) - 1
    lengths = ends - starts + 1
    return np.repeat(starts, lengths), np.repeat(ends, lengths)


def states(frames: pd.DataFrame) -> np.ndarray:
    """[frames, 21]: position, smoothed velocity, smoothed torque of the 7 motors."""
    first, _ = episode_bounds(frames)
    cuts = np.unique(first)[1:]
    velocity = np.concatenate([smooth_velocity(part) for part in np.split(np.stack(frames[OBS_VELOCITY]), cuts)])
    torque = np.concatenate(
        [
            np.stack([smooth_gripper_torque(joint) for joint in part.T], axis=1)
            for part in np.split(np.stack(frames[OBS_EFFORT]).astype(np.float64), cuts)
        ]
    )
    return np.concatenate([np.stack(frames[OBS_STATE]), velocity, torque], axis=1).astype(np.float32)


def windows(values: np.ndarray, frames: pd.DataFrame, offsets) -> np.ndarray:
    """[frames, len(offsets), width]: ``values`` at each frame + offset, held at the ends of the episode."""
    first, last = episode_bounds(frames)
    index = np.arange(len(values))[:, None] + np.asarray(offsets)[None]
    return values[np.clip(index, first[:, None], last[:, None])]


def command_targets(commands: np.ndarray, frames: pd.DataFrame, absolute_dims) -> np.ndarray:
    """[frames, 7]: the command minus the one before it; ``absolute_dims`` keep the command itself."""
    targets = commands - windows(commands, frames, [-1])[:, 0]
    targets[:, absolute_dims] = commands[:, absolute_dims]
    return targets


def quantiles(values: np.ndarray) -> dict[str, list[float]]:
    return {"q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist()}


def normalization_stats(roots, absolute_dims) -> dict[str, dict[str, list[float]]]:
    """Pooled 1% / 99% quantiles of the 21-wide state and of the command targets."""
    tables = [read_frames(root) for root in roots]
    return {
        "state": quantiles(np.concatenate([states(t) for t in tables])),
        "action": quantiles(
            np.concatenate([command_targets(np.stack(t[ACTION]), t, absolute_dims) for t in tables])
        ),
    }


def mistakes(root, frames: int) -> np.ndarray:
    """[frames] bool from meta/mistakes.parquet (a mistake within the next second, annotation_rubric.md)."""
    flags = np.zeros(frames, dtype=bool)
    for row in pd.read_parquet(Path(root) / "meta/mistakes.parquet").itertuples():
        flags[row.from_index : row.to_index] = True
    return flags


def speed_levels(root, frames: int) -> np.ndarray:
    """[frames] index into SPEED_LEVELS from meta/speed_level.parquet; -1 = no level."""
    levels = np.full(frames, -1)
    for row in pd.read_parquet(Path(root) / "meta/speed_level.parquet").itertuples():
        levels[row.from_index : row.to_index] = SPEED_LEVELS.index(row.level)
    return levels
