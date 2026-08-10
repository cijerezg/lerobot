#!/usr/bin/env python
r"""Measure the hold-predictor error the relative trajectory metrics divide by, and
report the floors that cap how much weight one chunk may carry.

``trajectory_error_components`` reports each error relative to the same error committed
by ``hold`` — the chunk that repeats the measured state and never moves. That denominator
depends on the demonstrated actions and the measured state alone, never on a prediction,
so it is measurable straight from a dataset with no model and no probe run.

Relative error is the raw error under a per-chunk weight $w = 1/d$, so a chunk the
demonstration barely moves through would dominate a batch. Flooring $d$ caps that weight
at $1/d_{\min}$ while leaving every chunk above the floor exactly unbiased; the floor
printed here is the corpus median over ``--max-weight-ratio``. Copy the reported values
into ``DEFAULT_SCALE_FLOORS`` in ``lerobot/utils/action_metrics.py``.

Normalization is affine per dim, so the additive term cancels from every difference below
and only the gain survives. QUANTILES maps x to 2(x - q01)/(q99 - q01) - 1, hence a gain
of 2/(q99 - q01); MEAN_STD maps x to (x - mean)/std, hence 1/std.
"""

import argparse
import glob
import logging

import numpy as np
import pandas as pd
import torch

from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

QUANTILES = [0.1, 1, 5, 25, 50, 75, 95, 99]


def normalization_gain(stats: dict, mode: str) -> np.ndarray:
    """Per-dim multiplier the normalizer applies, the only part that survives a difference."""
    if mode == "quantiles":
        return 2.0 / (np.asarray(stats["q99"], dtype=np.float64) - np.asarray(stats["q01"], dtype=np.float64))
    return 1.0 / np.asarray(stats["std"], dtype=np.float64)


def hold_denominators(
    roots: list[str], chunk_size: int, encoding: str, gain: np.ndarray
) -> dict[str, np.ndarray]:
    """Path, shape, and terminal error of the hold predictor, one row per chunk."""
    collected: dict[str, list[np.ndarray]] = {"path": [], "shape": [], "terminal": []}
    episodes = 0
    for root in roots:
        files = sorted(glob.glob(f"{root}/data/**/*.parquet", recursive=True))
        if not files:
            raise ValueError(f"no data parquet under {root}")
        for file in files:
            frame = pd.read_parquet(file, columns=["episode_index", "frame_index", ACTION, OBS_STATE])
            for _, episode in frame.groupby("episode_index"):
                episode = episode.sort_values("frame_index")
                actions = np.stack(episode[ACTION].to_numpy()).astype(np.float64)
                states = np.stack(episode[OBS_STATE].to_numpy()).astype(np.float64)
                # Only chunks lying wholly inside the episode: a padded tail is masked out
                # of the loss anyway, and would otherwise report motion that never happened.
                count = len(actions) - chunk_size
                if count <= 0:
                    continue
                episodes += 1
                window = np.lib.stride_tricks.sliding_window_view(actions, chunk_size, axis=0)[:count]
                window = np.transpose(window, (0, 2, 1))  # [chunk, time, dim]

                if encoding == "absolute":
                    target, hold = window, states[:count, None, :]
                else:
                    # Under anchor/delta the hold chunk is the zero vector, so the whole
                    # denominator is the demonstration's own encoded excursion.
                    target = window - states[:count, None, :]
                    hold = np.zeros_like(target)
                difference = (target - hold) * gain

                collected["path"].append((difference**2).mean(axis=(1, 2)))
                collected["terminal"].append((difference[:, -1, :] ** 2).mean(axis=1))
                collected["shape"].append((np.diff(difference, axis=1) ** 2).mean(axis=(1, 2)))

    logger.info(f"{episodes} episodes across {len(roots)} root(s)")
    return {name: np.concatenate(values) for name, values in collected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, nargs="+", help="Dataset root(s)")
    parser.add_argument(
        "--stats", type=str, required=True, help="Action stats .pt from compute_delta_stats.py"
    )
    parser.add_argument("--chunk-size", type=int, default=30, help="Action horizon")
    parser.add_argument("--encoding", type=str, required=True, choices=["absolute", "anchor", "delta"])
    parser.add_argument("--normalization", type=str, default="quantiles", choices=["quantiles", "mean_std"])
    parser.add_argument(
        "--max-weight-ratio",
        type=float,
        default=10.0,
        help="Most gradient weight one chunk may carry relative to a typical one; the floor is median / this.",
    )
    args = parser.parse_args()

    stats = torch.load(args.stats, weights_only=False)
    if stats.get("encoding") != args.encoding or stats.get("chunk_size") != args.chunk_size:
        raise ValueError(
            f"stats were built for encoding={stats.get('encoding')} chunk_size={stats.get('chunk_size')}, "
            f"but --encoding {args.encoding} --chunk-size {args.chunk_size} was requested"
        )

    denominators = hold_denominators(
        args.root, args.chunk_size, args.encoding, normalization_gain(stats, args.normalization)
    )
    chunks = len(next(iter(denominators.values())))
    logger.info(f"{chunks} chunks\n")

    header = "".join(f"{f'p{q:g}':>12}" for q in QUANTILES)
    print(f"{'denominator':<12}{header}{'min':>12}{'max':>12}")
    for name, values in denominators.items():
        row = "".join(f"{np.percentile(values, q):>12.3e}" for q in QUANTILES)
        print(f"{name:<12}{row}{values.min():>12.3e}{values.max():>12.3e}")

    print(f"\nDEFAULT_SCALE_FLOORS at --max-weight-ratio {args.max_weight_ratio:g}:")
    for name, values in denominators.items():
        floor = float(np.median(values)) / args.max_weight_ratio
        binds = float((values < floor).mean())
        unfloored = float(values.max() / max(values.min(), 1e-30))
        print(
            f'    "{name}": {floor:.1e},'
            f"   # binds on {binds:.2%} of chunks; caps a weight spread of {unfloored:.3g}x"
        )


if __name__ == "__main__":
    main()
