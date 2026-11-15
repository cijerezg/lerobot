"""Motor velocity as prompt tokens (docs/design/memory_prompts.md §1.1).

The same smoothing runs at buffer load (whole episodes) and in the actor (the last
control steps). The processor turns the smoothed velocity into state tokens.
"""

import numpy as np

# Weights of the causal mean, newest frame first.
VELOCITY_WEIGHTS = (0.25, 0.25, 0.2, 0.15, 0.15)
VELOCITY_SUFFIX = ".vel"
# Smoothed velocity per frame and joint in deg/s; NaN = no reading (the clause is left out).
VELOCITY_KEY = "motor_velocity"


def smooth_velocity(velocity: np.ndarray) -> np.ndarray:
    """Weighted mean of each frame and the 4 before it, for one episode [frames, joints] in time order.

    The first frames repeat the episode's first sample.
    """
    lookback = len(VELOCITY_WEIGHTS) - 1
    padded = np.concatenate([np.repeat(velocity[:1], lookback, axis=0), velocity])
    smoothed = np.zeros(velocity.shape)
    for age, weight in enumerate(VELOCITY_WEIGHTS):
        smoothed += weight * padded[lookback - age : len(padded) - age]
    return smoothed
