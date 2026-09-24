"""The low-pass filter the robot's commands actually pass through.

This is the last transform between a policy's absolute action chunk and the
controller, so anything that claims to show or score what the arm would do has to
apply it: the RTC runtime (`rl/rtc_actor_runtime.py`) and the probe adapters both
call it. The implementation stays independent of the hardware rollout stack.

Order matters. The filter is linear and runs on the *absolute* chunk, after
anchor/delta reconstruction — under delta encoding, filtering the increments and
then integrating is a different trajectory. The safety bounds (`bound_action_chunk`)
come after it. The RTC runtime applies them through `bound_policy_actions`, but the
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
    lag_limits=None,
    delta_limits=None,
    clamp_limits=None,
    step_limits=None,
) -> torch.Tensor:
    """Bound an absolute [T, D] chunk in robot units, relative to ``anchor`` (the
    observed state the chunk was inferred from, shape [D]). Four stages, each skipped
    when its limit is None:

    1. lag:       |a_0 - anchor| <= lag_limits[j]        (tick 0 only)
    2. excursion: |a_t - anchor| <= delta_limits[j]
    3. absolute:  clamp_limits[j][0] <= a_t <= clamp_limits[j][1]
    4. rate:      a_t <- a_{t-1} + clip(a_t - a_{t-1}, -step_limits[j], step_limits[j])
                  for t >= 1, chained from the bounded a_0

    The demos' a_0 lags s_0 by the follower's tracking error (q99 ~ 18 deg on the
    shoulder), so the first tick gets its own measured bound and the rate stage
    measures what its limit was measured on: consecutive commands. The absolute
    clamp is a contraction, so it cannot undo the lag or excursion bounds.
    """
    anchor = anchor.to(actions)
    if lag_limits is not None:
        limit = torch.as_tensor(lag_limits, dtype=actions.dtype, device=actions.device)
        actions = torch.cat([anchor + (actions[0] - anchor).clamp(-limit, limit)[None], actions[1:]])
    if delta_limits is not None:
        limit = torch.as_tensor(delta_limits, dtype=actions.dtype, device=actions.device)
        actions = anchor + (actions - anchor).clamp(-limit, limit)
    if clamp_limits is not None:
        limit = torch.as_tensor(clamp_limits, dtype=actions.dtype, device=actions.device)
        actions = actions.clamp(limit[:, 0], limit[:, 1])
    if step_limits is not None:
        limit = torch.as_tensor(step_limits, dtype=actions.dtype, device=actions.device)
        bounded = torch.empty_like(actions)
        previous = actions[0]
        bounded[0] = previous
        for t in range(1, actions.shape[0]):
            previous = previous + (actions[t] - previous).clamp(-limit, limit)
            bounded[t] = previous
        actions = bounded
    return actions
