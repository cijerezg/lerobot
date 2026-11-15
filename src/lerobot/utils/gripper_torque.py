"""Gripper torque as a prompt level (docs/design/memory_prompts.md §1.1).

The same two functions run at buffer load (whole episodes) and in the actor (the last
control steps), so the level in the prompt is computed the same way in both places.
"""

import numpy as np

# Weights of the causal mean, newest frame first.
GRIPPER_TORQUE_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
GRIPPER_TORQUE_KEY = "gripper.torque"
# Prompt level per frame, 1..levels; 0 = no reading (the clause is left out).
GRIPPER_TORQUE_LEVEL_KEY = "gripper_torque_level"


def smooth_gripper_torque(torque: np.ndarray) -> np.ndarray:
    """Weighted mean of each frame and the 3 before it, for one episode in time order.

    The first frames repeat the episode's first sample.
    """
    lookback = len(GRIPPER_TORQUE_WEIGHTS) - 1
    padded = np.concatenate([np.full(lookback, torque[0]), torque])
    smoothed = np.zeros(len(torque))
    for age, weight in enumerate(GRIPPER_TORQUE_WEIGHTS):
        smoothed += weight * padded[lookback - age : len(padded) - age]
    return smoothed


def gripper_torque_levels(smoothed: np.ndarray, levels: int, low: float, high: float) -> np.ndarray:
    """Equal-width levels 1..levels between low and high (N*m); values outside go to the end levels."""
    level = np.floor((smoothed - low) / (high - low) * levels).astype(np.int64) + 1
    return np.clip(level, 1, levels)
