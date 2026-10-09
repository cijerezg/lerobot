#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Per-action-layout normalization stats for the mixed ReBot + diverse run.

``compute_embodiment_stats.py`` groups roots by robot. That is not fine enough here:
DROID and FMB are both a Franka Panda, but DROID commands joint targets while FMB
copies measured joints, and RoboChallenge's gripper is a width in metres while UR7e's
is a ratio in [0, 1]. Pooling two such conventions into one row makes q01/q99 span the
union and squashes both, which is the same failure that motivated per-embodiment stats
one level up.

So rows are keyed by ``action_layout_id`` (lerobot/datasets/diverse_actor_selection.py),
and the artifact says so in ``stats_index_key`` -- a run cannot pair layout-keyed
statistics with embodiment indices by accident.

Statistics are computed over the ACTUAL training selection: every retained anchor of
every accepted episode, on full ``(30, D)`` chunks, in each source's native width. Only
real dimensions contribute; padding is applied after the statistics, never before.

Rows for layouts that contributed no data (e.g. a layout registered for a source this
run does not use) are filled with the pooled row over the layouts that did, and flagged
in ``row_has_data``. Those rows are never gathered by this run's samples, and the flag
makes an accidental gather auditable rather than invisible.

Example:
    python -m lerobot.scripts.compute_diverse_stats \\
        --encoding anchor --chunk-size 30 --name diverse-rebot-v1 \\
        --diverse-root outputs/diverse_robot_dataset \\
        --rebot-root outputs/rebot_socks_basket-annotated-v2 \\
        --rebot-root outputs/rebot_shirts_bin-annotated-v2 \\
        --out outputs/stats/action_stats_anchor_diverse-rebot-v1.pt
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.diverse_actor_selection import (
    ACTION_LAYOUTS,
    action_layout_by_name,
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.rl.data_sources.diverse_actor_buffer import DiverseActorBuffer, DiverseSampleSpec
from lerobot.scripts.compute_embodiment_stats import (
    ARTIFACT_FORMAT,
    LAYOUT,
    QUANTILES,
    _encode,
    _stats_for,
    collect_root,
)
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

STATS_INDEX_KEY = "action_layout_id"


def collect_diverse(root: str | Path, chunk_size: int, encoding: str) -> dict[int, dict[str, Any]]:
    """Encoded chunks and states per action layout, over the whole training selection."""
    selection = select_actor_anchors(open_federated_corpus(root))
    buffer = DiverseActorBuffer(selection, DiverseSampleSpec(load_images=False, load_depth=False))
    actions: dict[int, list[np.ndarray]] = defaultdict(list)
    states: dict[int, list[np.ndarray]] = defaultdict(list)

    for index, row in enumerate(selection.rows):
        try:
            sample = buffer.load_sample(index)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load actor row {index}: {row['episode_id']} anchor "
                f"{row['anchor_s']}s (frame {row['anchor_frame']}, source {row['source']})"
            ) from exc
        chunk = np.asarray(sample["action"], dtype=np.float32)
        if chunk.shape[0] != chunk_size:
            raise ValueError(
                f"{row['episode_id']} packs {chunk.shape[0]}-point chunks but --chunk-size is "
                f"{chunk_size}; the stats must match cfg.policy.chunk_size."
            )
        layout_id = int(row["action_layout_id"])
        actions[layout_id].append(chunk)
        states[layout_id].append(np.asarray(sample["state"], dtype=np.float32))

    collected: dict[int, dict[str, Any]] = {}
    for layout_id in sorted(actions):
        chunk_array = np.stack(actions[layout_id])
        state_array = np.stack(states[layout_id])
        collected[layout_id] = {
            "action": _encode(chunk_array, state_array, encoding),
            "state": state_array,
            "action_dim": int(chunk_array.shape[-1]),
            "state_dim": int(state_array.shape[-1]),
            "sources": [f"diverse:{root}"],
            # Absolute chunks and their anchors, for the hand block (--hand).
            "raw_action": chunk_array,
            "anchor": state_array,
        }
        logger.info(
            "  layout %d %-38s %6d chunks  D=%d",
            layout_id,
            ACTION_LAYOUTS[layout_id].name,
            len(chunk_array),
            chunk_array.shape[-1],
        )
    return collected


def collect_rebot(
    roots: list[Path], chunk_size: int, encoding: str, layout_name: str
) -> dict[int, dict[str, Any]]:
    """The ReBot half, as one layout row assembled from its LeRobotDataset roots."""
    if not roots:
        return {}
    layout = action_layout_by_name(layout_name)
    chunk_groups, state_groups, raw_groups = [], [], []
    for root in roots:
        chunks, states = collect_root(root, chunk_size, encoding)
        if len(chunks):
            chunk_groups.append(chunks)
            raw_groups.append(collect_root(root, chunk_size, "absolute")[0])
        state_groups.append(states)
    if not chunk_groups:
        raise ValueError(f"ReBot roots {roots} yielded no action chunks.")
    action = np.concatenate(chunk_groups, axis=0)
    state = np.concatenate(state_groups, axis=0)
    raw_action = np.concatenate(raw_groups, axis=0)
    if encoding != "anchor":
        raise ValueError(
            "collect_rebot recovers chunk anchors from the anchor encoding; use --encoding anchor."
        )
    anchor = raw_action[:, 0] - action[:, 0]
    if action.shape[-1] != layout.dim or state.shape[-1] != layout.dim:
        raise ValueError(
            f"layout {layout.name} declares {layout.dim}D but the ReBot roots store "
            f"action {action.shape[-1]}D / state {state.shape[-1]}D."
        )
    logger.info("  layout %d %-38s %6d chunks  D=%d", layout.index, layout.name, len(action), layout.dim)
    return {
        layout.index: {
            "action": action,
            "state": state,
            "action_dim": layout.dim,
            "state_dim": layout.dim,
            "sources": [str(root) for root in roots],
            "raw_action": raw_action,
            "anchor": anchor,
        }
    }


# q01/q99 band half-width in block scales. The QUANTILES normalizer maps [q01, q99] to [-1, 1]
# and the clamp step cuts there, so the band must hold almost every value: for the joints the
# empirical q01/q99 clip 2 %, and a Gaussian's q01/q99 sit at +-2.326 sigma, so the same band
# in block scales gives the hand block the joints' clip rate and the joints' chance-MSE (~0.18).
HAND_BAND_SCALES = 2.326


def _block_stats(values: np.ndarray, blocks: tuple[tuple[int, int], ...]) -> dict[str, np.ndarray]:
    """Global stats of a hand array ([N, T, D] deltas or [M, D] states): per-coordinate means,
    one scale per block (sqrt of the mean variance over the block's coordinates, per step), and
    q01/q99 = mean -/+ HAND_BAND_SCALES * scale, so the normalizer standardizes every coordinate
    of a block by that one scale (TODO.md, "Units")."""
    mean = values.mean(axis=0)
    var = values.var(axis=0)
    scale = np.ones_like(mean)
    for lo, hi in blocks:
        scale[..., lo:hi] = np.sqrt(var[..., lo:hi].mean(axis=-1, keepdims=True))
    stats = {
        "min": values.min(axis=0).astype(np.float32),
        "max": values.max(axis=0).astype(np.float32),
        "mean": mean.astype(np.float32),
        "std": scale.astype(np.float32),
    }
    for q in QUANTILES:
        stats[f"q{int(round(q * 100)):02d}"] = np.quantile(values, q, axis=0).astype(np.float32)
    stats["q01"] = (mean - HAND_BAND_SCALES * scale).astype(np.float32)
    stats["q99"] = (mean + HAND_BAND_SCALES * scale).astype(np.float32)
    for lo, hi in blocks:
        clipped = (
            (values[..., lo:hi] < stats["q01"][..., lo:hi]) | (values[..., lo:hi] > stats["q99"][..., lo:hi])
        ).mean()
        logger.info("    block %d:%d clipped by the q01/q99 band: %.2f %%", lo, hi, 100 * clipped)
    return stats


def collect_hand(collected: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Hand deltas of every chunk and hand states of every state, pooled over the layouts
    that have a kinematic chain (policies/molmoact2/hand_block.LAYOUT_CHAIN_KEYS)."""
    from lerobot.policies.molmoact2.hand_block import (
        HAND_DELTA_BLOCKS,
        HAND_STATE_BLOCKS,
        hand_delta,
        hand_pose,
        hand_state,
        layout_has_hand,
    )

    deltas, states, rows = [], [], {}
    for layout_id, entry in sorted(collected.items()):
        if not layout_has_hand(layout_id):
            continue
        ids = torch.full((len(entry["anchor"]),), layout_id, dtype=torch.long)
        p0, r0, g0, _ = hand_pose(ids, torch.from_numpy(entry["anchor"]).to(torch.float64))
        pt, rt, gt, _ = hand_pose(ids, torch.from_numpy(entry["raw_action"]).to(torch.float64))
        delta = hand_delta(p0[:, None], r0[:, None], g0[:, None], pt, rt, gt).numpy()
        ids_s = torch.full((len(entry["state"]),), layout_id, dtype=torch.long)
        ps, rs, gs, _ = hand_pose(ids_s, torch.from_numpy(entry["state"]).to(torch.float64))
        hand_states = hand_state(ps, rs, gs).numpy()
        deltas.append(delta)
        states.append(hand_states)
        rows[ACTION_LAYOUTS[layout_id].name] = {"chunks": int(len(delta)), "states": int(len(hand_states))}
        logger.info(
            "  hand layout %d %-38s %6d chunks: |dp| p99 %.3f m, |drot| p99 %.3f rad, |dg| p99 %.3f m",
            layout_id,
            ACTION_LAYOUTS[layout_id].name,
            len(delta),
            np.percentile(np.linalg.norm(delta[..., :3], axis=-1), 99),
            np.percentile(np.linalg.norm(delta[..., 3:6], axis=-1), 99),
            np.percentile(np.abs(delta[..., 6]), 99),
        )
    if not deltas:
        raise ValueError("--hand: no collected layout has a kinematic chain.")
    delta_array = np.concatenate(deltas)
    state_array = np.concatenate(states)
    return {
        "action": _block_stats(delta_array, HAND_DELTA_BLOCKS),
        "state": _block_stats(state_array, HAND_STATE_BLOCKS),
        "rows": rows,
        "chunks": int(len(delta_array)),
        "states": int(len(state_array)),
    }


def build_artifact(
    collected: dict[int, dict[str, Any]], chunk_size: int, encoding: str, hand: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not collected:
        raise ValueError("no layouts collected; nothing to build stats from.")
    action_width = max(entry["action_dim"] for entry in collected.values())
    state_width = max(entry["state_dim"] for entry in collected.values())
    num_rows = max(collected) + 1
    if hand is not None:
        from lerobot.policies.molmoact2.hand_block import (
            HAND_ACTION_WIDTH,
            HAND_STATE_WIDTH,
            JOINT_SLOTS,
            layout_has_hand,
        )

        if action_width > JOINT_SLOTS or state_width > JOINT_SLOTS:
            raise ValueError(
                f"--hand expects joint widths <= {JOINT_SLOTS}, got {action_width}/{state_width}."
            )
        action_width, state_width = JOINT_SLOTS, JOINT_SLOTS

    # The fallback row for a layout this run does not use: pooled over everything that
    # did contribute. It is never gathered here; it exists so an unused row is a sane
    # identity-ish transform rather than a NaN or a divide by zero.
    pooled_action = np.concatenate(
        [
            np.pad(e["action"], ((0, 0), (0, 0), (0, action_width - e["action_dim"])))
            for e in collected.values()
        ]
    )
    pooled_state = np.concatenate(
        [np.pad(e["state"], ((0, 0), (0, state_width - e["state_dim"]))) for e in collected.values()]
    )

    action_stats: dict[str, list[np.ndarray]] = defaultdict(list)
    state_stats: dict[str, list[np.ndarray]] = defaultdict(list)
    action_mask, state_mask, counts, row_names, row_has_data = [], [], [], [], []
    native_action_dims, native_state_dims, sources = [], [], {}

    for row in range(num_rows):
        layout = ACTION_LAYOUTS[row]
        entry = collected.get(row)
        has_data = entry is not None
        if not has_data:
            entry = {
                "action": pooled_action,
                "state": pooled_state,
                "action_dim": action_width,
                "state_dim": state_width,
                "sources": [],
            }
        for stat, value in _stats_for(entry["action"], action_width).items():
            action_stats[stat].append(value)
        for stat, value in _stats_for(entry["state"], state_width).items():
            state_stats[stat].append(value)
        action_mask.append([index < entry["action_dim"] for index in range(action_width)])
        state_mask.append([index < entry["state_dim"] for index in range(state_width)])
        counts.append([len(entry["action"]), len(entry["state"])] if has_data else [0, 0])
        row_names.append(layout.name)
        row_has_data.append(has_data)
        native_action_dims.append(entry["action_dim"])
        native_state_dims.append(entry["state_dim"])
        sources[layout.name] = entry["sources"]

    def stack(group: dict[str, list[np.ndarray]], mask: list[list[bool]]) -> dict[str, torch.Tensor]:
        out = {stat: torch.from_numpy(np.stack(values)) for stat, values in group.items()}
        out["mask"] = torch.tensor(mask, dtype=torch.bool)
        return out

    hand_payload = None
    if hand is not None:
        # The hand columns are global: the same numbers in every row, valid only for rows
        # whose layout has a chain. native_action_dims stay the joint widths; the hand step
        # flips the hand slots of the padding masks at run time.
        has_hand = [layout_has_hand(row) and row in collected for row in range(num_rows)]
        for stat in list(action_stats):
            action_stats[stat] = [
                np.concatenate([value, hand["action"][stat]], axis=-1) for value in action_stats[stat]
            ]
        for stat in list(state_stats):
            state_stats[stat] = [
                np.concatenate([value, hand["state"][stat]], axis=-1) for value in state_stats[stat]
            ]
        action_mask = [
            row + [has] * (HAND_ACTION_WIDTH - JOINT_SLOTS)
            for row, has in zip(action_mask, has_hand, strict=True)
        ]
        state_mask = [
            row + [has] * (HAND_STATE_WIDTH - JOINT_SLOTS)
            for row, has in zip(state_mask, has_hand, strict=True)
        ]
        action_width, state_width = HAND_ACTION_WIDTH, HAND_STATE_WIDTH
        hand_payload = {
            "action_slots": [JOINT_SLOTS, HAND_ACTION_WIDTH],
            "state_slots": [JOINT_SLOTS, HAND_STATE_WIDTH],
            "layout_has_hand": has_hand,
            "delta_block_scale": torch.from_numpy(hand["action"]["std"]),
            "state_block_scale": torch.from_numpy(hand["state"]["std"]),
            "rows": hand["rows"],
            "chunks": hand["chunks"],
            "states": hand["states"],
        }

    return {
        "hand_block": hand_payload,
        "format": ARTIFACT_FORMAT,
        "layout": LAYOUT,
        "encoding": encoding,
        "chunk_size": chunk_size,
        "action_width": action_width,
        "state_width": state_width,
        "stats_index_key": STATS_INDEX_KEY,
        "row_names": row_names,
        # Kept so a reader that predates stats_index_key still finds a vocabulary; the
        # names are layouts, which is exactly why stats_index_key must be honoured.
        "embodiments": row_names,
        "row_has_data": row_has_data,
        "native_action_dims": native_action_dims,
        "native_state_dims": native_state_dims,
        "counts": torch.tensor(counts, dtype=torch.long),
        "roots": sources,
        "stats": {
            ACTION: stack(action_stats, action_mask),
            OBS_STATE: stack(state_stats, state_mask),
        },
    }


def report(artifact: dict[str, Any]) -> None:
    logger.info(
        "\n%d rows, widths a%d/s%d, %s encoding, chunk_size %d",
        len(artifact["row_names"]),
        artifact["action_width"],
        artifact["state_width"],
        artifact["encoding"],
        artifact["chunk_size"],
    )
    for row, name in enumerate(artifact["row_names"]):
        counts = artifact["counts"][row].tolist()
        flag = "" if artifact["row_has_data"][row] else "   (no data -- pooled fallback)"
        q01 = artifact["stats"][ACTION]["q01"][row]
        q99 = artifact["stats"][ACTION]["q99"][row]
        span = (q99 - q01).flatten()
        logger.info(
            "  %d %-38s chunks=%-7d states=%-7d action q99-q01 span mean=%.4f%s",
            row,
            name,
            counts[0],
            counts[1],
            float(span.mean()),
            flag,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--diverse-root", default="outputs/diverse_robot_dataset")
    parser.add_argument("--rebot-root", action="append", default=[], type=Path)
    parser.add_argument("--rebot-layout", default="rebot_b601_joint7_commanded")
    parser.add_argument("--encoding", default="anchor", choices=("absolute", "anchor", "delta"))
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument(
        "--hand",
        action="store_true",
        help="append the hand block columns (fingertip pose from the kinematics assets); "
        "action width 15, state width 18, see policies/molmoact2/hand_block.py",
    )
    parser.add_argument("--name", default="diverse-rebot-v1")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-diverse", action="store_true")
    args = parser.parse_args()

    collected: dict[int, dict[str, Any]] = {}
    if not args.skip_diverse:
        logger.info("Diverse corpus:")
        collected.update(collect_diverse(args.diverse_root, args.chunk_size, args.encoding))
    if args.rebot_root:
        logger.info("ReBot roots:")
        collected.update(collect_rebot(args.rebot_root, args.chunk_size, args.encoding, args.rebot_layout))

    hand = collect_hand(collected) if args.hand else None
    artifact = build_artifact(collected, args.chunk_size, args.encoding, hand)
    report(artifact)

    out = Path(args.out or f"outputs/stats/action_stats_{args.encoding}_{args.name}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out)
    logger.info("\nWrote %s", out)


if __name__ == "__main__":
    main()
