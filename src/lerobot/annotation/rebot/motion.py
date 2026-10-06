"""Proprio events of a ReBot episode: gripper closes, still stretches, idle runs, leaving home.

One definition of each, used by every pass. State is [N, 7] in degrees, gripper last (0 = shut).
"""

import numpy as np


def runs_of(mask):
    """[(start, end)) of every run of True."""
    e = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(int), 0]))
    return list(zip(e[::2], e[1::2], strict=True))


def closed_intervals(g, travel=25.0, persist=15):
    """Gripper-closed intervals by the depth-label v2 relative-travel rule: closed at +travel deg over the
    running minimum, open again at -travel below the running maximum; shorter than persist frames dropped."""
    out, closed, start, lo, hi = [], False, None, g[0], g[0]
    for i, v in enumerate(g):
        if not closed:
            lo = min(lo, v)
            if v >= lo + travel:
                closed, start, hi = True, i, v
        else:
            hi = max(hi, v)
            if v <= hi - travel:
                closed, lo = False, v
                if i - start >= persist:
                    out.append((start, i))
    if closed and len(g) - start >= persist:
        out.append((start, len(g)))
    return out


def still_runs(state, win=30):
    """Stretches for review shading: windows with < 1 deg total motion per arm joint and < 1 % of the
    episode's gripper span."""
    arm = state[:, :6]
    span = np.ptp(state[:, 6]) + 1e-9
    n = len(state)
    still = np.zeros(n, bool)
    for i in range(n - win):
        seg = arm[i : i + win]
        if np.abs(np.diff(seg, axis=0)).sum(0).max() < 1.0 and np.ptp(state[i : i + win, 6]) < 0.01 * span:
            still[i : i + win] = True
    return runs_of(still)


def idle_runs(state, a, b, home=None, win=30, min_s=3.0, thr=2.0, fps=30):
    """Idle runs inside [a, b), the rule that decides cuts (user 2026-10-04): every joint and the gripper
    within thr deg over every win-frame window, for at least min_s seconds. ``home`` (bool[b - a]) marks
    "return to home" frames, where only the arm joints count."""
    s, n = state[a:b], b - a
    still = np.zeros(n, bool)
    for t in range(n - win + 1):
        w = s[t : t + win]
        rng = w.max(0) - w.min(0)
        if home is not None and home[t : t + win].all():
            rng = rng[:6]
        if rng.max() < thr:
            still[t : t + win] = True
    return [[a + p, a + q] for p, q in runs_of(still) if q - p >= min_s * fps]


def idle_cut(run, keep, margin=15):
    """The cut for one idle run: keep margin frames each side; a run touching the head or the tail of the
    kept range is trimmed to it. Both ends stay on the depth phase (multiples of 3 from the range start)."""
    (f0, f1), (a, b) = run, keep
    c0 = a if f0 == a else f0 + margin + (-(f0 + margin - a)) % 3
    c1 = b if f1 == b and f0 != a else f1 - margin
    return [c0, c1 - (c1 - c0) % 3]


def leaves_home(state, thr=10.0):
    """First frame where one of the first four joints is more than thr deg from its start (-1 if never)."""
    init = np.median(state[:30, :4], axis=0)
    on = np.flatnonzero(np.abs(state[:, :4] - init).max(1) > thr)
    return int(on[0]) if len(on) else -1
