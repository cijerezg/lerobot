"""Class-independent motion speed from measured arm joint states.

Speed at transition t is the L2 norm of (q[t+1]-q[t])*native_rate_hz,
smoothed over approximately 0.2 seconds. Fit 20/40/60/80 percentile edges
from reviewed training transitions within each robot group. Segment labels are
medians of these speeds; task quality, verbs, net displacement and release
inheritance do not enter the calculation. Gripper coordinates must be excluded
by the caller because their units differ from the arm joint angles.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

METHOD = 'joint_motion_v1'
SMOOTH_SECONDS = 0.2
QUANTILES = (0.2, 0.4, 0.6, 0.8)


def motion_trace(q: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return speed and validity for transitions t -> t+1, in radians/second."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or not np.isfinite(rate) or rate <= 0:
        raise ValueError('Expected a 2-D joint array and finite positive native rate')
    raw = np.linalg.norm(np.diff(q, axis=0), axis=1) * rate
    valid = np.isfinite(raw)
    if not len(raw):
        return raw, valid
    window = max(1, round(SMOOTH_SECONDS * rate))
    smooth = uniform_filter1d(np.where(valid, raw, 0.0), size=window, mode='nearest')
    # A nonfinite observation makes its entire smoothing neighborhood unscorable.
    valid = minimum_filter1d(valid.astype(np.uint8), size=window, mode='nearest').astype(bool)
    smooth = np.maximum(smooth, 0.0)
    smooth[~valid] = np.nan
    return smooth, valid


def fit_group(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    edges = np.quantile(values, QUANTILES).tolist() if len(values) else []
    usable = len(values) >= 100 and len(edges) == 4 and all(b > a for a, b in zip(edges, edges[1:]))
    return {'n_training_transitions': int(len(values)), 'edges_rad_s': edges,
            'usable': usable, 'median_rad_s': float(np.median(values)) if len(values) else None}


def bucket(value: float, cell: dict) -> int:
    if not cell['usable'] or not np.isfinite(value):
        return 3
    return 1 + int(np.searchsorted(cell['edges_rad_s'], value, side='right'))


def summarize(values: np.ndarray, valid: np.ndarray, cell: dict) -> dict:
    """A chunk is a display/storage summary of its states, not a reference class."""
    values, valid = np.asarray(values), np.asarray(valid)
    reason = ''
    if len(values) < 2:
        reason = 'fewer than two state transitions in the atom'
    elif not valid.all():
        reason = 'nonfinite state data in the smoothing window'
    elif not cell['usable']:
        reason = 'insufficient or degenerate robot motion reference'
    good = values[valid & np.isfinite(values)]
    if len(good):
        p10, median, p90 = np.quantile(good, [0.1, 0.5, 0.9]).tolist()
        mean = float(np.mean(good))
    else:
        p10 = median = p90 = mean = None
    return {'speed': 3 if reason else bucket(median, cell),
            'speed_source': 'default_unclear' if reason else METHOD,
            'speed_default_reason': reason,
            'motion_median_rad_s': median, 'motion_mean_rad_s': mean,
            'motion_p10_rad_s': p10, 'motion_p90_rad_s': p90,
            'state_transition_count': int(len(values))}
