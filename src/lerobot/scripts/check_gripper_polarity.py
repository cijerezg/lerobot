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

"""Which end of the gripper channel means "closed", measured per dataset root.

Source cards disagree, and one of them refuses to say. DROID states its gripper is
"absolute in [0,1], 0 fully open and 1 fully closed". The UR7e card gives a [0,1] ratio
and says the open/closed orientation "is not explicitly stated ... and must not be
inferred or changed". So in a cross-embodiment corpus the same slot may point opposite
ways on two robots, and nothing downstream can tell: per-embodiment normalization fixes
scale, and anchor/delta encoding removes offset, but neither touches SIGN. A grasp on
one robot and a grasp on another then look like opposite commands.

This measures the convention instead of assuming it, and never rewrites anything — the
output is provenance, which is not the silent relabeling the UR7e card warns against.

Two estimators vote, and they are the ones grounded in what the robot was doing:

  grasp   Net change across subtasks whose text describes closing ("grasp", "pick up").
          Should move toward closed.
  release Net change across subtasks whose text describes opening ("release", "place").
          Should move toward open — the opposite sign to `grasp`, which also controls
          for any channel-wide drift that would fool one span type alone.

A third signal, `start` (which end of the range an episode opens at), is REPORTED BUT
DOES NOT VOTE. It rests on the gripper being open at the start of an episode, and the
rebot B601 is a counterexample: it rests closed at ~0 and opens toward -270, so the
prior calls it exactly backwards. It is kept only as context for reading the range.

Agreement between the voting estimators is the result worth trusting; disagreement is
reported as disagreement rather than resolved into a false verdict. A root with no
usable spans reports "no signal" rather than falling back to a guess.

Example:
    python lerobot/src/lerobot/scripts/check_gripper_polarity.py \\
        --root outputs/diverse_robot_dataset/corpus/packed_5hz/droid/AUTOLab \\
        --root outputs/diverse_robot_dataset/corpus/packed_5hz/ur7e/stack_block
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.embodiment import canonical_embodiment
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

SUBTASK_KEY = "annotation.subtask"

# Verbs describing the gripper closing on something, and letting it go. Matched on word
# boundaries so "unplace" or "graspable" cannot sneak in. A span matching BOTH classes
# ("carry the block to the dish and release it") is dropped rather than guessed at.
CLOSE_CUES = (
    r"grasp", r"grab", r"pick up", r"pick the", r"picks up", r"take the",
    r"close the gripper", r"clamp", r"pinch",
)
OPEN_CUES = (
    r"release", r"let go", r"drop", r"place", r"put down", r"set (it|them|the)",
    r"open the gripper", r"deposit",
)


def _read_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text())


def _stack(column: Any) -> np.ndarray:
    rows = list(column)
    try:
        stacked = np.stack(rows)
        if stacked.dtype != object:
            return stacked.astype(np.float32)
    except (ValueError, TypeError):
        pass
    return np.asarray([np.asarray(np.asarray(r).tolist(), dtype=np.float32) for r in rows])


def _load(root: Path) -> pd.DataFrame | None:
    info = _read_info(root)
    available = set((info.get("features") or {}).keys())
    wanted = [OBS_STATE, "episode_index"]
    for optional in (ACTION, SUBTASK_KEY, "index", "frame_index"):
        if optional in available:
            wanted.append(optional)
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        logger.warning("  %s: no data/**/*.parquet", root.name)
        return None
    frames = []
    for path in files:
        try:
            frames.append(pq.read_table(path, columns=wanted).to_pandas())
        except Exception as error:  # a component still being written by `build_corpus pack`
            logger.warning("  %s: skipping unreadable %s (%s)", root.name, path.name, type(error).__name__)
            return None
    table = pd.concat(frames, ignore_index=True)
    order = "index" if "index" in table.columns else ("frame_index" if "frame_index" in table.columns else None)
    if order is not None:
        table = table.sort_values(order, kind="stable").reset_index(drop=True)
    return table


def _classify(text: str) -> str | None:
    """"close", "open", or None when the span says both or neither."""
    low = str(text or "").lower()
    closes = any(re.search(rf"\b{cue}", low) for cue in CLOSE_CUES)
    opens = any(re.search(rf"\b{cue}", low) for cue in OPEN_CUES)
    if closes and not opens:
        return "close"
    if opens and not closes:
        return "open"
    return None


def _windows_from_meta(root: Path) -> list[tuple[int, int, str]]:
    """Spans from meta/subtask_windows.json, for roots with no per-frame label column.

    Entries hold global [from_index, to_index) dataset frame ranges, the same convention
    offline_dataset_utils._subtask_indices_from_windows reads them under.
    """
    path = root / "meta" / "subtask_windows.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, dict):
        return []
    spans: list[tuple[int, int, str]] = []
    for windows in episodes.values():
        for window in windows or []:
            text = window.get("subtask")
            if text is None:
                continue
            spans.append((int(window["from_index"]), int(window["to_index"]), str(text)))
    return spans


def _span_changes(table: pd.DataFrame, gripper: np.ndarray, root: Path) -> dict[str, list[float]]:
    """Net gripper change across each annotated span, bucketed by what the span describes."""
    changes: dict[str, list[float]] = {"close": [], "open": []}

    if SUBTASK_KEY not in table.columns:
        for start, end, text in _windows_from_meta(root):
            end = min(end, len(gripper))
            if end - start < 2:
                continue
            kind = _classify(text)
            if kind is not None:
                changes[kind].append(float(gripper[end - 1] - gripper[start]))
        return changes
    labels = table[SUBTASK_KEY].astype(str).to_numpy()
    episodes = table["episode_index"].to_numpy()
    # A span is a maximal run of identical (episode, subtask) in row order.
    boundary = np.ones(len(table), dtype=bool)
    boundary[1:] = (labels[1:] != labels[:-1]) | (episodes[1:] != episodes[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], len(table))
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        kind = _classify(labels[start])
        if kind is None:
            continue
        changes[kind].append(float(gripper[end - 1] - gripper[start]))
    return changes


def _start_estimate(table: pd.DataFrame, gripper: np.ndarray, opening_frames: int) -> float | None:
    """Mean of each episode's opening frames, rescaled to where it sits in the channel range.

    0 means the episode starts at the channel's low end, 1 at the high end. The end it
    starts nearest is "open"; a value near 0.5 means the prior does not separate them.
    """
    # Robust ends: the rebot gripper reaches -714 on a handful of frames, and min/max
    # would let that one glitch define the whole range.
    low, high = float(np.nanquantile(gripper, 0.01)), float(np.nanquantile(gripper, 0.99))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-9:
        return None
    firsts = []
    for _, rows in table.groupby("episode_index", sort=False):
        head = gripper[rows.index.to_numpy()[:opening_frames]]
        if len(head):
            firsts.append(float(np.nanmean(head)))
    if not firsts:
        return None
    return (float(np.mean(firsts)) - low) / (high - low)


def _vote(value: float | None, positive_means: str, negative_means: str, deadzone: float) -> str | None:
    if value is None or abs(value) < deadzone:
        return None
    return positive_means if value > 0 else negative_means


def analyze(root: Path, gripper_index: int, opening_frames: int, deadzone: float) -> dict[str, Any] | None:
    table = _load(root)
    if table is None:
        return None
    states = _stack(table[OBS_STATE].to_numpy())
    if states.ndim != 2:
        logger.warning("  %s: unexpected state shape %s", root.name, states.shape)
        return None
    gripper = states[:, gripper_index]
    native_dim = states.shape[1]

    changes = _span_changes(table, gripper, root)
    grasp_mean = float(np.mean(changes["close"])) if changes["close"] else None
    release_mean = float(np.mean(changes["open"])) if changes["open"] else None
    start_position = _start_estimate(table, gripper, opening_frames)

    # A rise during a grasp means high = closed. A rise during a release means high = open.
    votes = {
        "grasp": _vote(grasp_mean, "high=closed", "high=open", deadzone),
        "release": _vote(release_mean, "high=open", "high=closed", deadzone),
    }
    cast = [v for v in votes.values() if v is not None]
    verdict = "no signal"
    if cast:
        verdict = max(set(cast), key=cast.count) if len(set(cast)) == 1 else "DISAGREEMENT"

    return {
        "root": root,
        "robot": canonical_embodiment(_read_info(root).get("robot_type")) or _read_info(root).get("robot_type"),
        "native_dim": native_dim,
        "rows": len(table),
        # Robust ends, so one -714 glitch frame cannot define the printed range.
        "range": (float(np.nanquantile(gripper, 0.01)), float(np.nanquantile(gripper, 0.99))),
        "raw_range": (float(np.nanmin(gripper)), float(np.nanmax(gripper))),
        "grasp_mean": grasp_mean,
        "grasp_n": len(changes["close"]),
        "release_mean": release_mean,
        "release_n": len(changes["open"]),
        "start_position": start_position,
        "votes": votes,
        "verdict": verdict,
    }


def _fmt(value: float | None, spec: str = "+.4f") -> str:
    return "     -" if value is None else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, required=True, action="append", help="Dataset root; repeatable.")
    parser.add_argument(
        "--gripper-index",
        type=int,
        default=-1,
        help="Gripper channel in observation.state. Default -1 (last), which matches every "
             "source card in the corpus: joints first, gripper last.",
    )
    parser.add_argument("--opening-frames", type=int, default=3, help="Frames per episode for the start prior.")
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.02,
        help="Mean |change| below this abstains instead of voting, so a channel that barely "
             "moves cannot cast a confident vote from noise.",
    )
    args = parser.parse_args()

    results = []
    for raw in args.root:
        root = Path(raw).expanduser()
        logger.info("[%s]", root)
        result = analyze(root, args.gripper_index, args.opening_frames, args.deadzone)
        if result is not None:
            results.append(result)
    if not results:
        logger.info("\nNothing readable.")
        return

    logger.info(
        "\n%-34s %-13s %4s %-15s %8s %5s %8s %5s %7s",
        "root", "robot", "dim", "q01..q99 range", "grasp", "n", "release", "n", "start",
    )
    for r in results:
        logger.info(
            "%-34s %-13s %4d [%6.2f,%6.2f] %8s %5d %8s %5d %7s",
            r["root"].name, str(r["robot"])[:13], r["native_dim"],
            r["range"][0], r["range"][1],
            _fmt(r["grasp_mean"]), r["grasp_n"],
            _fmt(r["release_mean"]), r["release_n"],
            _fmt(r["start_position"], ".2f"),
        )

    logger.info("\n%-34s %-14s %-14s  %s", "root", "grasp", "release", "verdict")
    for r in results:
        v = r["votes"]
        logger.info(
            "%-34s %-14s %-14s  %s",
            r["root"].name, v["grasp"] or "-", v["release"] or "-", r["verdict"],
        )

    verdicts = {r["verdict"] for r in results if r["verdict"] not in ("no signal", "DISAGREEMENT")}
    logger.info("")
    if any(r["verdict"] == "DISAGREEMENT" for r in results):
        logger.info("Estimators disagree on at least one root — inspect those before trusting any verdict.")
    elif len(verdicts) > 1:
        logger.info(
            "MISMATCH: roots disagree on convention (%s). The gripper slot points opposite ways "
            "across these robots; record the convention per source before training on them together.",
            ", ".join(sorted(verdicts)),
        )
    elif len(verdicts) == 1:
        logger.info("All roots agree: %s. No gripper sign conflict.", verdicts.pop())
    else:
        logger.info("No root produced a usable signal.")


if __name__ == "__main__":
    main()
