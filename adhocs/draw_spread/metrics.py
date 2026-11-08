"""Metrics for the draw-spread study (train vs val, 64 draws per anchor).

Space: the normalized, anchor-encoded action space the flow loss uses. A chunk is [T, D]
(T = 30 steps, D = 7 joints); draws are [K, T, D]. Every chunk splits exactly into
    offset  = its per-joint mean over the T steps            [D]
    shape   = the chunk minus that mean                       [T, D]
and for the mean square over (t, d):  msq(e) = msq_offset(e) + msq_shape(e).
Every metric below comes in the three columns total / offset / shape.

Per anchor, with a = demo chunk, X_k = draws, m = mean_k X_k, mu / var = cross-episode
reference at the anchor's (subtask, progress bin) from the whole training corpus:
    bias2        msq(m - a)                        where the policy's mean sits vs the demo
    spread       mean_k msq(X_k - m)               entropy proxy of the draw distribution
    mse          mean_k msq(X_k - a) = bias2 + spread
    hold         msq(a)                            motion size; the hold-still baseline (0 in anchor space)
    R            mse / hold                        0 exact, 1 no better than holding still
    var_data     var                               cross-episode variance of the demo chunks in the bucket
    spread_ratio spread / var_data                 ~1 calibrated, << 1 collapsed
    style_proj   <m - mu, a - mu> / <a - mu, a - mu>  1 = the mean reproduces this episode's deviation from
                                                      the cross-episode mean, 0 = it outputs the bucket mean
    style_cos    cosine of the same two vectors
    energy       mean_k rms(X_k - a) - 0.5 mean_{k,k'} rms(X_k - X_k')   proper score of the whole draw set
Under an input-noise setting j (vs the clean setting 0):
    shift        msq(m_j - m_0)                    how far the mean moves
    spread_rel   spread_j / spread_0               how the spread reacts
"""
from __future__ import annotations

import numpy as np

COMPONENTS = ("total", "offset", "shape")


def split(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """offset [..., D], shape [..., T, D] of chunks [..., T, D]."""
    offset = x.mean(axis=-2)
    return offset, x - offset[..., None, :]


def msq3(x: np.ndarray) -> np.ndarray:
    """Mean square over (t, d) of chunks [..., T, D] as the three components, stacked last: [..., 3]."""
    offset, shape = split(x)
    off = np.mean(np.square(offset), axis=-1)
    shp = np.mean(np.square(shape), axis=(-2, -1))
    return np.stack([off + shp, off, shp], axis=-1)


def inner3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """<x, y> per component (mean over (t, d), same normalization as msq3): [..., 3]."""
    xo, xs = split(x)
    yo, ys = split(y)
    off = np.mean(xo * yo, axis=-1)
    shp = np.mean(xs * ys, axis=(-2, -1))
    return np.stack([off + shp, off, shp], axis=-1)


def energy_score(draws: np.ndarray, gt: np.ndarray) -> float:
    """draws [K, T, D], gt [T, D]; rms distance over (t, d)."""
    k = draws.shape[0]
    to_gt = np.sqrt(np.mean(np.square(draws - gt), axis=(1, 2))).mean()
    diff = draws[:, None] - draws[None]
    pair = np.sqrt(np.mean(np.square(diff), axis=(2, 3)))
    pair_mean = pair.sum() / (k * (k - 1))
    return float(to_gt - 0.5 * pair_mean)


def anchor_metrics(draws: np.ndarray, gt: np.ndarray, ref_mean: np.ndarray, ref_var3: np.ndarray) -> dict[str, float]:
    """Scalar metrics of one anchor. draws [K, T, D], gt [T, D], ref_mean [T, D], ref_var3 [3]."""
    m = draws.mean(axis=0)
    bias2 = msq3(m - gt)
    spread = msq3(draws - m).mean(axis=0)
    mse = msq3(draws - gt).mean(axis=0)
    hold = msq3(gt)
    delta = gt - ref_mean
    rho = m - ref_mean
    dd = inner3(delta, delta)
    rr = inner3(rho, rho)
    proj = inner3(rho, delta) / np.maximum(dd, 1e-12)
    cos = inner3(rho, delta) / np.maximum(np.sqrt(dd * rr), 1e-12)
    out: dict[str, float] = {}
    for i, c in enumerate(COMPONENTS):
        out[f"bias2_{c}"] = float(bias2[i])
        out[f"spread_{c}"] = float(spread[i])
        out[f"mse_{c}"] = float(mse[i])
        out[f"hold_{c}"] = float(hold[i])
        out[f"R_{c}"] = float(mse[i] / max(hold[i], 1e-12))
        out[f"var_data_{c}"] = float(ref_var3[i])
        out[f"spread_ratio_{c}"] = float(spread[i] / max(ref_var3[i], 1e-12))
        out[f"style_proj_{c}"] = float(proj[i])
        out[f"style_cos_{c}"] = float(cos[i])
    out["energy"] = energy_score(draws, gt)
    return out


def noise_metrics(draws_noisy: np.ndarray, draws_clean: np.ndarray) -> dict[str, float]:
    m1, m0 = draws_noisy.mean(axis=0), draws_clean.mean(axis=0)
    shift = msq3(m1 - m0)
    s1 = msq3(draws_noisy - m1).mean(axis=0)
    s0 = msq3(draws_clean - m0).mean(axis=0)
    out = {}
    for i, c in enumerate(COMPONENTS):
        out[f"shift_{c}"] = float(shift[i])
        out[f"spread_rel_{c}"] = float(s1[i] / max(s0[i], 1e-12))
    return out


def bucket_reference(chunks: np.ndarray, keys: list[tuple]) -> dict[tuple, tuple[np.ndarray, np.ndarray, int]]:
    """Cross-episode reference per bucket key: (mean chunk [T, D], var3 [3], n). chunks [N, T, D]."""
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    out = {}
    for key, idx in groups.items():
        sub = chunks[idx]
        mean = sub.mean(axis=0)
        var3 = msq3(sub - mean).mean(axis=0) if len(idx) > 1 else np.full(3, np.nan)
        out[key] = (mean, var3, len(idx))
    return out
