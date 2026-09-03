#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-embodiment normalization stats for MolmoAct2.

``compute_delta_stats.py`` pools every root into ONE set of action quantiles, and
``pool_lowdim_stats`` does the same for state. That is correct only while every root
is the same robot: a Franka's joint ranges and an ARX5's are different distributions,
and pooling them makes q01/q99 span the union, so each robot's own motion gets
squashed into a narrow part of [-1, 1] before the 256-bin state discretization ever
sees it.

This script instead groups the roots BY EMBODIMENT and pools within each group, then
stacks the groups into one artifact indexed by the embodiment vocabulary in
``lerobot.datasets.embodiment``. The normalizer gathers a row per sample using the
``embodiment_index`` column the replay buffer carries.

Robots differ in DOF (Franka 8, ARX5/UR5/UR7e/rebot 7), so every group is padded to a
common width and carries a boolean ``mask`` marking its real dims. That mask is the
same one the masked normalizer already honours, so padded dims pass through untouched
instead of being normalized against meaningless stats.

Two on-disk layouts are handled, distinguished by the shape of the ``action`` feature
in meta/info.json:

  * flat (``action`` is ``[D]``) — one action per row, as the rebot roots store it.
    Chunks are assembled from episode boundaries, never across them.
  * chunked (``action`` is ``[T, D]``) — the whole horizon is already materialized per
    row, as the diverse corpus packed components store it. Rows are used directly,
    which is also why those roots need no episode metadata.

Example:
    python lerobot/src/lerobot/scripts/compute_embodiment_stats.py \\
        --encoding anchor --chunk-size 30 --name diverse-v1 \\
        --root outputs/rebot_socks_basket-annotated-v2 \\
        --root outputs/diverse_robot_dataset/corpus/packed_5hz/robochallenge/arrange_flowers
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from lerobot.datasets.embodiment import EMBODIMENT_NAMES, canonical_embodiment
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

ARTIFACT_FORMAT = "embodiment_stats_v2"
LAYOUT = "native_source_order_v1"
ENCODINGS = ("absolute", "anchor", "delta")
# Matches lerobot.datasets.compute_stats.DEFAULT_QUANTILES so a per-embodiment artifact
# is a drop-in for the pooled stats it replaces.
QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
# The (low, high) pairs a normalizer divides by: QUANTILES mode uses q01/q99,
# QUANTILE10 uses q10/q90. Both need the degenerate-band fallback in _stats_for.
_QUANTILE_BANDS = (("q01", "q99"), ("q10", "q90"))


def _read_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{root} has no meta/info.json; cannot resolve its embodiment or layout.")
    return json.loads(info_path.read_text())


def _feature_shape(info: dict[str, Any], key: str) -> list[int]:
    feature = (info.get("features") or {}).get(key)
    if not isinstance(feature, dict) or not feature.get("shape"):
        raise ValueError(f"meta/info.json has no shape for feature {key!r}.")
    return [int(dim) for dim in feature["shape"]]


def _stack_column(column: Any) -> np.ndarray:
    """Stack a parquet column of per-row arrays into one dense float32 array.

    A 1-D feature arrives as a fixed_size_list and stacks directly. A multi-dim
    feature (the packed corpus stores ``action`` as ``list<list<double>>``) arrives as
    a per-row object array of arrays, which numpy will not stack without being walked
    down to nested lists first.
    """
    rows = list(column)
    try:
        stacked = np.stack(rows)
        if stacked.dtype != object:
            return stacked.astype(np.float32)
    except (ValueError, TypeError):
        pass
    return np.asarray([np.asarray(np.asarray(row).tolist(), dtype=np.float32) for row in rows])


def _restore_shape(values: np.ndarray, declared: list[int], key: str, root: Path) -> np.ndarray:
    """Reshape a column to its declared per-row shape.

    v3.0 roots normally store a multi-dim feature as a nested list, which stacks
    straight to [N, *shape]. A root that flattened it instead stacks to [N, prod(shape)],
    so restore the declared shape rather than silently treating 210 numbers as 210 dims.
    """
    expected = tuple(declared)
    if values.shape[1:] == expected:
        return values
    if int(np.prod(values.shape[1:])) != int(np.prod(expected)):
        raise ValueError(
            f"{root} column {key!r} has per-row shape {values.shape[1:]}, which does not "
            f"match the declared {expected} in meta/info.json."
        )
    return values.reshape(len(values), *expected)


def _read_columns(root: Path, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Stacked (action, state) arrays for a root, in dataset row and source order.

    Nothing is permuted here: the pipeline normalizes vectors in the order the corpus
    recorded them, so the statistics have to describe those same slots.
    """
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root} has no data/**/*.parquet.")
    frames = [pq.read_table(f, columns=[ACTION, OBS_STATE]).to_pandas() for f in files]
    table = pd.concat(frames, ignore_index=True)
    actions = _stack_column(table[ACTION].to_numpy())
    states = _stack_column(table[OBS_STATE].to_numpy())
    actions = _restore_shape(actions, _feature_shape(info, ACTION), ACTION, root)
    states = _restore_shape(states, _feature_shape(info, OBS_STATE), OBS_STATE, root)
    return actions.astype(np.float32), states.astype(np.float32)


def _encode(targets: np.ndarray, anchors: np.ndarray, encoding: str) -> np.ndarray:
    """Encode [N, T, D] absolute targets against [N, D] anchors.

    Mirrors anchor_encoding._encode exactly — the stats must describe the same
    quantity the pipeline normalizes, or the quantiles are meaningless.
    """
    if encoding == "absolute":
        return targets
    if encoding == "anchor":
        return targets - anchors[:, None, :]
    first = targets[:, :1, :] - anchors[:, None, :]
    if targets.shape[1] == 1:
        return first
    return np.concatenate([first, np.diff(targets, axis=1)], axis=1)


def _chunks_from_flat_root(
    root: Path, info: dict[str, Any], chunk_size: int, encoding: str
) -> np.ndarray:
    """Assemble [N, T, D] chunks from a one-action-per-row root, never crossing episodes."""
    meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not meta_files:
        raise FileNotFoundError(
            f"{root} stores one action per row, so chunks must come from episode "
            "boundaries, but it has no meta/episodes/*.parquet."
        )
    episodes = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
    actions, states = _read_columns(root, info)

    chunks: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    for _, episode in episodes.iterrows():
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        ep_actions = actions[start:end]
        ep_states = states[start:end]
        valid_starts = len(ep_actions) - chunk_size + 1
        if valid_starts <= 0:
            continue
        for i in range(valid_starts):
            chunks.append(ep_actions[i : i + chunk_size])
            anchors.append(ep_states[i])
    if not chunks:
        return np.empty((0, chunk_size, actions.shape[-1]), dtype=np.float32)
    return _encode(np.stack(chunks), np.stack(anchors), encoding)


def _chunks_from_chunked_root(
    root: Path, info: dict[str, Any], chunk_size: int, encoding: str
) -> np.ndarray:
    """Use the already-materialized [T, D] action chunk on each row."""
    actions, states = _read_columns(root, info)
    if actions.ndim != 3:
        raise ValueError(f"{root} declares a chunked action but its column is {actions.shape}.")
    if actions.shape[1] != chunk_size:
        raise ValueError(
            f"{root} stores chunks of {actions.shape[1]} steps but --chunk-size is {chunk_size}. "
            "The stats must match cfg.policy.chunk_size or make_processors hard-errors."
        )
    return _encode(actions, states, encoding)


def collect_root(root: Path, chunk_size: int, encoding: str) -> tuple[np.ndarray, np.ndarray]:
    """(encoded action chunks [N, T, D], raw states [M, D]) for one root."""
    info = _read_info(root)
    action_shape = _feature_shape(info, ACTION)
    _, states = _read_columns(root, info)
    if len(action_shape) == 1:
        chunks = _chunks_from_flat_root(root, info, chunk_size, encoding)
        layout = "flat"
    else:
        chunks = _chunks_from_chunked_root(root, info, chunk_size, encoding)
        layout = "chunked"
    logger.info(
        "    %-58s %-8s %6d chunks  %6d states  D=%d",
        root.name,
        layout,
        len(chunks),
        len(states),
        states.shape[-1],
    )
    return chunks, states


def _stats_for(values: np.ndarray, width: int) -> dict[str, np.ndarray]:
    """min/max/mean/std/quantiles over axis 0, right-padded to `width` on the last axis.

    Padding uses 0 for locations and 1 for scales so a padded dim, if it is ever
    normalized despite the mask, is an identity transform rather than a divide by zero.
    """
    native = values.shape[-1]
    trailing = values.shape[1:-1]

    def pad(array: np.ndarray, fill: float) -> np.ndarray:
        out = np.full((*trailing, width), fill, dtype=np.float32)
        out[..., :native] = array
        return out

    stats: dict[str, np.ndarray] = {
        "min": pad(values.min(axis=0), 0.0),
        "max": pad(values.max(axis=0), 1.0),
        "mean": pad(values.mean(axis=0), 0.0),
        "std": pad(values.std(axis=0), 1.0),
    }
    for q in QUANTILES:
        # q01/q99 straddle 0..1 on padded dims so their denominator is 1, not 0.
        fill = 0.0 if q < 0.5 else 1.0
        stats[f"q{int(round(q * 100)):02d}"] = pad(np.quantile(values, q, axis=0), fill)

    # A quantile band that is zero, or numerically indistinguishable from it, carries no
    # scale. The normalizer only guards denom == 0 exactly, so such a band divides by its
    # eps and pushes every real sample past the [-1, 1] clamp, collapsing distinct actions
    # onto one target. FMB's gripper did this: >99% of chunks have no gripper motion in the
    # first ~200 ms, so q01 == q99 == 0 at small k, and "closing" normalized to the same -1
    # as "holding". min/max is then the only scale left. Where the data really is constant
    # -- a copy_state layout's k=0 anchor delta, or a padded dim -- max == min and this is
    # a no-op, leaving the existing eps path to produce its constant.
    extent = stats["max"] - stats["min"]
    for lo, hi in _QUANTILE_BANDS:
        if lo not in stats or hi not in stats:
            continue
        collapsed = (stats[hi] - stats[lo] <= 1e-9 * extent) & (extent > 0)
        stats[lo] = np.where(collapsed, stats["min"], stats[lo])
        stats[hi] = np.where(collapsed, stats["max"], stats[hi])
    return stats


def build_artifact(
    roots: list[Path],
    chunk_size: int,
    encoding: str,
    overrides: dict[str, str],
) -> dict[str, Any]:
    by_embodiment: dict[str, list[Path]] = defaultdict(list)
    for root in roots:
        override = overrides.get(str(root))
        raw = override or _read_info(root).get("robot_type")
        name = canonical_embodiment(raw)
        if name is None:
            raise ValueError(
                f"{root} has robot_type={raw!r}, which is not a known embodiment. "
                "Add it to EMBODIMENT_NAMES/EMBODIMENT_ALIASES in lerobot/datasets/embodiment.py, "
                "or pass --embodiment <root>=<name>."
            )
        by_embodiment[name].append(root)

    collected: dict[str, dict[str, Any]] = {}
    for name in sorted(by_embodiment):
        logger.info("[%s]", name)
        chunk_groups, state_groups = [], []
        for root in by_embodiment[name]:
            chunks, states = collect_root(root, chunk_size, encoding)
            if len(chunks):
                chunk_groups.append(chunks)
            state_groups.append(states)
        if not chunk_groups:
            raise ValueError(f"Embodiment {name!r} yielded no action chunks from {by_embodiment[name]}.")
        action_dims = {group.shape[-1] for group in chunk_groups}
        state_dims = {group.shape[-1] for group in state_groups}
        if len(action_dims) != 1 or len(state_dims) != 1:
            raise ValueError(
                f"Embodiment {name!r} has inconsistent action widths {sorted(action_dims)} "
                f"or state widths {sorted(state_dims)} across roots."
            )
        collected[name] = {
            "action": np.concatenate(chunk_groups, axis=0),
            "state": np.concatenate(state_groups, axis=0),
            "action_dim": action_dims.pop(),
            "state_dim": state_dims.pop(),
            "roots": [str(root) for root in by_embodiment[name]],
        }

    action_width = max(entry["action_dim"] for entry in collected.values())
    state_width = max(entry["state_dim"] for entry in collected.values())
    names = [name for name in EMBODIMENT_NAMES if name in collected]
    logger.info(
        "\nPadding %d embodiment(s) to action/state widths %d/%d: %s",
        len(names),
        action_width,
        state_width,
        ", ".join(
            f"{name}(a{collected[name]['action_dim']}/s{collected[name]['state_dim']})"
            for name in names
        ),
    )

    action_stats: dict[str, list[np.ndarray]] = defaultdict(list)
    state_stats: dict[str, list[np.ndarray]] = defaultdict(list)
    action_mask, state_mask, counts, native_action_dims, native_state_dims, roots_by_name = (
        [],
        [],
        [],
        [],
        [],
        {},
    )
    for name in names:
        entry = collected[name]
        for stat, value in _stats_for(entry["action"], action_width).items():
            action_stats[stat].append(value)
        for stat, value in _stats_for(entry["state"], state_width).items():
            state_stats[stat].append(value)
        action_mask.append([index < entry["action_dim"] for index in range(action_width)])
        state_mask.append([index < entry["state_dim"] for index in range(state_width)])
        counts.append([len(entry["action"]), len(entry["state"])])
        native_action_dims.append(entry["action_dim"])
        native_state_dims.append(entry["state_dim"])
        roots_by_name[name] = entry["roots"]

    def stack(group: dict[str, list[np.ndarray]], mask: list[list[bool]]) -> dict[str, torch.Tensor]:
        out = {stat: torch.from_numpy(np.stack(values)) for stat, values in group.items()}
        out["mask"] = torch.tensor(mask, dtype=torch.bool)
        return out

    return {
        "format": ARTIFACT_FORMAT,
        "layout": LAYOUT,
        "encoding": encoding,
        "chunk_size": chunk_size,
        "action_width": action_width,
        "state_width": state_width,
        "embodiments": names,
        "native_action_dims": native_action_dims,
        "native_state_dims": native_state_dims,
        "counts": torch.tensor(counts, dtype=torch.long),
        "roots": roots_by_name,
        "stats": {
            ACTION: stack(action_stats, action_mask),
            OBS_STATE: stack(state_stats, state_mask),
        },
    }


def report(artifact: dict[str, Any]) -> None:
    """Print the per-embodiment q01/q99 spread, and what pooling them would have cost."""
    names = artifact["embodiments"]
    for key in (OBS_STATE, ACTION):
        stats = artifact["stats"][key]
        q01, q99 = stats["q01"], stats["q99"]
        if q01.ndim == 3:  # action: [E, T, D] -> collapse the horizon
            q01, q99 = q01.amin(dim=1), q99.amax(dim=1)
        logger.info("\n%s q01..q99 per dim", key)
        for row, name in enumerate(names):
            width = int(stats["mask"][row].sum())
            spans = " ".join(f"{q99[row, d] - q01[row, d]:6.2f}" for d in range(width))
            logger.info("  %-14s %s", name, spans)
        if len(names) > 1:
            pooled = q99.amax(dim=0) - q01.amin(dim=0)
            own = q99 - q01
            valid = stats["mask"]
            ratio = torch.where(valid, pooled[None, :] / own.clamp(min=1e-6), torch.nan)
            logger.info("  %-14s %s", "pooled/own", "(x wider each robot's range becomes if pooled)")
            for row, name in enumerate(names):
                width = int(valid[row].sum())
                spans = " ".join(f"{ratio[row, d]:6.2f}" for d in range(width))
                logger.info("  %-14s %s", name, spans)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, required=True, action="append", help="Dataset root; repeatable.")
    parser.add_argument("--encoding", type=str, required=True, choices=ENCODINGS)
    parser.add_argument("--chunk-size", type=int, required=True, help="Must match cfg.policy.chunk_size.")
    parser.add_argument("--name", type=str, required=True, help="Output basename.")
    parser.add_argument("--output-dir", type=str, default="outputs/stats")
    parser.add_argument(
        "--embodiment",
        type=str,
        action="append",
        default=[],
        metavar="ROOT=NAME",
        help="Override a root's embodiment when its meta/info.json robot_type is missing or wrong.",
    )
    args = parser.parse_args()

    overrides = {}
    for item in args.embodiment:
        if "=" not in item:
            parser.error(f"--embodiment expects ROOT=NAME, got {item!r}.")
        root, _, name = item.partition("=")
        overrides[root] = name

    roots = [Path(root).expanduser() for root in args.root]
    artifact = build_artifact(roots, args.chunk_size, args.encoding, overrides)
    report(artifact)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"embodiment_stats_{args.encoding}_{args.name}.pt"
    torch.save(artifact, out)
    logger.info("\nSaved %s", out)
    logger.info(
        "  %d embodiments x widths %s, %s encoding, chunk_size %d",
        len(artifact["embodiments"]),
        f"a{artifact['action_width']}/s{artifact['state_width']}",
        artifact["encoding"],
        artifact["chunk_size"],
    )


if __name__ == "__main__":
    main()
