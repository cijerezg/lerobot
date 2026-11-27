"""flux3_rl in the RTC runtime (rl/rtc_actor_runtime.py): the policy, the state window and the chunk decode."""

import numpy as np

from lerobot.policies.flux3 import Flux3Policy
from lerobot.utils.gripper_torque import smooth_gripper_torque
from lerobot.utils.motor_velocity import smooth_velocity

# Absolute commands before now, [B, n_obs_steps, 7]; the processor integrates the chunk's deltas from the last one.
COMMAND_HISTORY = "observation.command_history"


class Flux3RuntimePolicy(Flux3Policy):
    """The runtime's call: the sampler takes no correction from the last chunk and draws its own noise."""

    def predict_action_chunk(
        self, batch, inference_delay=None, prev_chunk_left_over=None, execution_horizon=None, generator=None
    ):
        return super().predict_action_chunk(batch)


def state_window(positions: np.ndarray, velocities, torques, offsets) -> np.ndarray:
    """[len(offsets), 21]: position, smoothed velocity, smoothed torque of the 7 motors at now + offsets.

    ``positions`` is the window itself. ``velocities`` and ``torques`` are the raw readings of the
    episode's last control steps, oldest first, the last one now; they are smoothed as in
    ``windows.states`` and held at the episode's first reading as in ``windows.windows``.
    """
    velocity = smooth_velocity(np.array(velocities))
    torque = np.stack([smooth_gripper_torque(joint) for joint in np.array(torques, dtype=np.float64).T], axis=1)
    index = np.clip(len(velocity) - 1 + np.asarray(offsets), 0, None)
    return np.concatenate([positions, velocity[index], torque[index]], axis=1).astype(np.float32)


def chunk_postprocessor(postprocessor):
    """The runtime decodes one chunk [T, 7]; the flux3 postprocessor reads 2 dimensions as one step per row."""
    return lambda chunk: postprocessor(chunk[None])[0]
