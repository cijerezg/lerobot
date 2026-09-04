"""The low-pass filter the robot's commands actually pass through.

This is the last transform between a policy's absolute action chunk and the
controller, so anything that claims to show or score what the arm would do has to
apply it: the deployed runtimes (`rl/inference_utils.py`, `rl/rtc_actor_runtime.py`)
and the probe adapters both call it. It lives here rather than next to the runtimes
because importing those pulls in matplotlib, OpenCV and the whole rollout stack.

Order matters. The filter is linear and runs on the *absolute* chunk, after
anchor/delta reconstruction — under delta encoding, filtering the increments and
then integrating is a different trajectory. The safety bounds (`bound_action_chunk`)
come after it. They live here so both runtimes share one implementation, but the
probes must not apply them: they are a guard on the robot, and folding them into a
measurement would hide the very violations a probe reports.
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


def bound_action_chunk(
    actions: torch.Tensor,
    anchor: torch.Tensor,
    delta_limits=None,
    clamp_limits=None,
    step_limits=None,
) -> torch.Tensor:
    """Bound an absolute [T, D] chunk in robot units, relative to ``anchor`` (the
    observed state the chunk was inferred from, shape [D]). Three stages, each skipped
    when its limit is None:

    1. excursion: |a_t - anchor| <= delta_limits[j]
    2. absolute:  clamp_limits[j][0] <= a_t <= clamp_limits[j][1]
    3. rate:      a_t <- a_{t-1} + clip(a_t - a_{t-1}, -step_limits[j], step_limits[j]),
                  with a_{-1} = anchor

    The absolute clamp is a contraction, so it cannot undo the excursion bound. The
    rate stage runs last and in tracking form, so an anchor outside the workspace
    walks to the box edge at step_limits per tick instead of jumping to it.
    """
    anchor = anchor.to(actions)
    if delta_limits is not None:
        limit = torch.as_tensor(delta_limits, dtype=actions.dtype, device=actions.device)
        actions = anchor + (actions - anchor).clamp(-limit, limit)
    if clamp_limits is not None:
        limit = torch.as_tensor(clamp_limits, dtype=actions.dtype, device=actions.device)
        actions = actions.clamp(limit[:, 0], limit[:, 1])
    if step_limits is not None:
        limit = torch.as_tensor(step_limits, dtype=actions.dtype, device=actions.device)
        bounded = torch.empty_like(actions)
        previous = anchor
        for t in range(actions.shape[0]):
            previous = previous + (actions[t] - previous).clamp(-limit, limit)
            bounded[t] = previous
        actions = bounded
    return actions
