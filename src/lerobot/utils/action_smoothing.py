"""The low-pass filter the robot's commands actually pass through.

This is the last transform between a policy's absolute action chunk and the
controller, so anything that claims to show or score what the arm would do has to
apply it: the deployed runtimes (`rl/inference_utils.py`, `rl/rtc_actor_runtime.py`)
and the probe adapters both call it. It lives here rather than next to the runtimes
because importing those pulls in matplotlib, OpenCV and the whole rollout stack.

Order matters. The filter is linear and runs on the *absolute* chunk, after
anchor/delta reconstruction — under delta encoding, filtering the increments and
then integrating is a different trajectory. The per-joint safety clamp comes after
it and is deliberately not part of this module: it is a guard on the robot, and
folding it into a measurement would hide the very violations a probe reports.
"""

import numpy as np
import torch
from scipy.signal import butter, filtfilt

_BUTTER_B, _BUTTER_A = butter(N=2, Wn=0.2, btype="low")


def apply_butterworth_filter(actions: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Zero-phase low-pass Butterworth filter along the time axis of an [T, D]
    action chunk, returning the same type it was given. Returns input unchanged
    when T is too short for filtfilt's default padlen (3 * max(len(a), len(b)) = 9).

    The runtimes hold the chunk as a tensor and the probes reduce it in float64
    numpy; the filter is the same either way, and a caller having to convert around
    it invites converting on only one of the two paths.
    """
    if actions.shape[0] <= 9:
        return actions
    if isinstance(actions, torch.Tensor):
        arr = actions.detach().to(torch.float32).cpu().numpy()
        smoothed = filtfilt(_BUTTER_B, _BUTTER_A, arr, axis=0)
        return torch.as_tensor(smoothed.copy(), dtype=actions.dtype, device=actions.device)
    return np.ascontiguousarray(filtfilt(_BUTTER_B, _BUTTER_A, actions, axis=0))
