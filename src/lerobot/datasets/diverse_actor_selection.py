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

"""Training selection over the federated diverse corpus (integration plan phase A).

The corpus reader answers "what was recorded". This module answers the separate
question "what does the actor run train on", and it is the only place that decision
is spelled out:

* every accepted episode, every retained 5 Hz anchor -- ``split`` is provenance and
  is never allowed to filter this run;
* one stable ``action_layout_id`` per source convention, because two robots that both
  report "Franka" do not share an action layout (DROID commands joints, FMB copies
  measured ones) and normalization statistics must key on the layout;
* one canonical camera-role name per recorded camera, so sources that spell the same
  physical view differently land in the same model slot and an unmapped camera is a
  hard error rather than a silently dropped view;
* the per-anchor ``mistake`` flag recomputed from the reviewed event spans.

The stored anchor flag is segment-level: it is true for every anchor inside a segment
that contains any mistake event, which over-claims on 65 common anchors whose own
instant sits outside the event. ReBot's metadata column is per-frame (a frame is a
mistake frame only inside a confirmed window; see ``ReplayBuffer.materialize_metadata``)
and DIVERSE_ROBOT_DATASET.md section 6 defines a mistake as an event "whose boolean is
true only over that span". A 50/50 ReBot + diverse mixture feeds one prompt clause, so
the two corpora have to mean the same thing by it.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.datasets.diverse_corpus import HISTORY_OFFSETS_S
from lerobot.datasets.fmb_corpus import FederatedDiverseCorpus

# ── Corpus ledger (DIVERSE_ROBOT_DATASET.md section 3.3) ──────────────────────
# The training selection is all of it. Startup asserts these, so a corpus that was
# rebuilt, half-copied, or filtered cannot quietly train on a different dataset.
EXPECTED_EPISODES = 404
EXPECTED_ANCHORS = 60_728
EXPECTED_EPISODES_BY_SOURCE = {
    "robochallenge": 200,
    "droid": 50,
    "droid_success": 50,
    "ur7e": 4,
    "fmb": 100,
}
EXPECTED_ANCHORS_BY_SOURCE = {
    "robochallenge": 40_706,
    "droid": 5_774,
    "droid_success": 5_823,
    "ur7e": 954,
    "fmb": 7_471,
}

# ── Packed observation window ────────────────────────────────────────────────
# Every actor sample carries seven observations at [-6, -5, -4, -3, -2, -1, 0] s.
# Training consumes four of them: three history slots plus the current frame. The
# slots are selected by age here rather than by hardcoded index, so changing
# memory.history_offsets_seconds moves them without touching any consumer.
PACKED_OFFSETS_S: tuple[float, ...] = tuple(float(offset) for offset in HISTORY_OFFSETS_S)
PACKED_OBSERVATIONS = len(PACKED_OFFSETS_S)
PACKED_CURRENT_SLOT = PACKED_OBSERVATIONS - 1

FUTURE_POINTS = 30
FUTURE_RATE_HZ = 30.0


def packed_slot_for_age(age_s: float, *, tolerance_s: float = 1e-6) -> int:
    """Index into the seven packed observations for a lookback age in seconds.

    ``age_s`` is a magnitude (6.0 means six seconds ago), matching
    ``MemoryConfig.history_times_seconds``. Negative values are accepted and read as
    the same instant, matching ``memory.history_offsets_seconds``.
    """
    wanted = -abs(float(age_s))
    for slot, offset in enumerate(PACKED_OFFSETS_S):
        if abs(offset - wanted) <= tolerance_s:
            return slot
    raise ValueError(
        f"lookback {age_s} s is not one of the packed observations "
        f"{list(PACKED_OFFSETS_S)}; the corpus stores no frame at that instant."
    )


def packed_history_slots(ages_s: list[float]) -> list[int]:
    """Packed indices for the configured history ages, oldest → newest.

    Pass ``MemoryConfig.history_times_seconds()`` (already normalized oldest → newest);
    the returned slots preserve that order, which is the order every consumer emits.
    """
    slots = [packed_slot_for_age(age) for age in ages_s]
    if len(set(slots)) != len(slots):
        raise ValueError(f"history ages {ages_s} collapse onto duplicate packed slots {slots}.")
    if PACKED_CURRENT_SLOT in slots:
        raise ValueError(
            f"history ages {ages_s} include the current observation (slot {PACKED_CURRENT_SLOT}); "
            "current state/RGB/depth travel through the ordinary current-observation keys."
        )
    return slots


# ── Camera roles ─────────────────────────────────────────────────────────────
# Three canonical slots. A source with two cameras leaves external_1 absent; absence
# is carried as a mask, never as a copied or black-filled view.
CANONICAL_CAMERA_ROLES: tuple[str, ...] = ("external_0", "external_1", "wrist_0")

# Recorded camera name → canonical role, per source. One map per source covers both
# RoboChallenge embodiments: UR5 simply has no "side" camera.
CAMERA_ROLE_MAP: dict[str, dict[str, str]] = {
    "droid": {"left_external": "external_0", "right_external": "external_1", "wrist": "wrist_0"},
    "droid_success": {"left_external": "external_0", "right_external": "external_1", "wrist": "wrist_0"},
    "fmb": {"side_1": "external_0", "side_2": "external_1", "wrist_1": "wrist_0"},
    "robochallenge": {"global": "external_0", "side": "external_1", "wrist": "wrist_0"},
    "ur7e": {"realsense_topview": "external_0", "realsense_wrist": "wrist_0"},
    # ReBot is not part of this corpus; its map lives here so the mixed run has one
    # camera vocabulary and the ReBot cache probe can check itself against it.
    "rebot": {"top": "external_0", "wrist": "wrist_0"},
}


def camera_roles_for(source: str, cameras: list[str]) -> dict[str, str]:
    """{canonical role: recorded camera} for one episode. Unmapped cameras raise."""
    mapping = CAMERA_ROLE_MAP.get(source)
    if mapping is None:
        raise KeyError(f"source {source!r} has no camera-role map; add one before training on it.")
    roles: dict[str, str] = {}
    for camera in cameras:
        role = mapping.get(camera)
        if role is None:
            raise KeyError(
                f"{source} camera {camera!r} is unmapped. Every recorded camera must land in a "
                f"canonical role ({', '.join(CANONICAL_CAMERA_ROLES)}) or be removed deliberately."
            )
        if role in roles:
            raise ValueError(f"{source} maps {roles[role]!r} and {camera!r} onto the same role {role!r}.")
        roles[role] = camera
    return roles


# ── Action layouts ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActionLayout:
    """One source's action/state convention, stable across runs.

    ``index`` is what a buffer column and a per-layout statistics artifact key on, so
    this tuple is APPEND-ONLY for the same reason ``EMBODIMENT_NAMES`` is: reordering
    it silently relabels every sample of every run trained before the change.

    ``embodiment`` is the broad robot name; it is deliberately not the key. DROID and
    FMB are both Franka Panda and share neither action semantics nor gripper units.
    """

    index: int
    name: str
    source: str
    embodiment: str
    dim: int
    action_source: str
    gripper: str


ACTION_LAYOUTS: tuple[ActionLayout, ...] = (
    ActionLayout(0, "droid_franka_joint8_commanded", "droid", "Franka", 8, "native", "command_0_open_1_closed"),
    ActionLayout(
        1, "droid_success_franka_joint8_commanded", "droid_success", "Franka", 8, "native",
        "command_0_open_1_closed",
    ),
    ActionLayout(2, "fmb_franka_joint8_measured", "fmb", "Franka", 8, "copy_state", "source_gripper_pose_levels"),
    ActionLayout(
        3, "robochallenge_arx5_joint7_measured", "robochallenge", "ARX5", 7, "copy_state", "width_metres"
    ),
    ActionLayout(
        4, "robochallenge_ur5_joint7_measured", "robochallenge", "UR5", 7, "copy_state", "width_metres"
    ),
    ActionLayout(5, "ur7e_joint7_commanded", "ur7e", "UR7e", 7, "native", "ratio_0_1"),
    # ReBot: the established half of the 50/50 mixture. Its buffer is a LeRobotDataset,
    # not this corpus, but it needs an id in the same space for per-layout stats.
    ActionLayout(6, "rebot_b601_joint7_commanded", "rebot", "Rebot B601", 7, "native", "ratio_0_1"),
)

_LAYOUT_BY_KEY = {(layout.source, layout.embodiment): layout for layout in ACTION_LAYOUTS}
_LAYOUT_BY_NAME = {layout.name: layout for layout in ACTION_LAYOUTS}
MAX_ACTION_DIM = max(layout.dim for layout in ACTION_LAYOUTS)


def action_layout_for(source: str, embodiment: str) -> ActionLayout:
    layout = _LAYOUT_BY_KEY.get((source, embodiment))
    if layout is None:
        raise KeyError(
            f"no action layout registered for source={source!r} embodiment={embodiment!r}. "
            "Append one to ACTION_LAYOUTS -- never reuse an existing index."
        )
    return layout


def action_layout_by_name(name: str) -> ActionLayout:
    return _LAYOUT_BY_NAME[name]


# ── Per-anchor mistake flag ──────────────────────────────────────────────────


def _common_anchor_mistake(record: dict[str, Any], anchor_s: float) -> bool | None:
    """True when the anchor instant itself lies inside a reviewed mistake event."""
    for segment in record["annotations"]["segments"]:
        if float(segment["start_s"]) <= anchor_s < float(segment["end_s"]):
            return any(
                float(event["start_s"]) <= anchor_s < float(event["end_s"])
                for event in segment.get("mistake_events") or []
            )
    return None


def _fmb_anchor_mistake(record: dict[str, Any], timestep: int) -> bool | None:
    for interval in record["primitive_intervals"]:
        if int(interval["start_timestep"]) <= timestep < int(interval["end_timestep_exclusive"]):
            return any(
                int(event["start_timestep"]) <= timestep < int(event["end_timestep_exclusive"])
                for event in interval.get("mistake_events") or []
            )
    return None


# ── Selection ────────────────────────────────────────────────────────────────


@dataclass
class DiverseActorSelection:
    """All accepted episodes, all retained 5 Hz anchors, with training identity attached.

    ``rows`` are copies of the corpus rows -- the corpus caches and reuses its own dicts,
    and nothing here may mutate what the reader hands back.
    """

    corpus: FederatedDiverseCorpus
    rows: list[dict[str, Any]]
    episode_records: dict[str, dict[str, Any]]
    mistake_flags_corrected: int = 0

    @property
    def episode_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.rows:
            seen.setdefault(row["episode_id"], None)
        return list(seen)

    def rows_by_episode(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            grouped[row["episode_id"]].append(row)
        return dict(grouped)


def _episode_record(corpus: FederatedDiverseCorpus, row: dict[str, Any]) -> dict[str, Any]:
    if row["corpus_key"] == "fmb":
        return corpus.fmb._records[str(row["episode_id"])]  # noqa: SLF001 - the reader exposes no accessor
    root = corpus.common.root / "episodes" / str(row["episode_id"]) / "episode.json"
    return json.loads(root.read_text(encoding="utf-8"))


def select_actor_anchors(
    corpus: FederatedDiverseCorpus,
    *,
    verify_counts: bool = True,
) -> DiverseActorSelection:
    """The training selection: ``actor_anchors(split=None, retained_only=True)``.

    ``split`` is never passed. It stays on every row as provenance and is reported by
    the audit, but no code path may filter on it: this run trains on all of it.
    """
    rows = corpus.actor_anchors(split=None, retained_only=True)

    records: dict[str, dict[str, Any]] = {}
    prepared: list[dict[str, Any]] = []
    corrected = 0
    for source_row in rows:
        row = dict(source_row)
        episode_id = str(row["episode_id"])
        if episode_id not in records:
            records[episode_id] = _episode_record(corpus, row)
        record = records[episode_id]

        layout = action_layout_for(row["source"], row["embodiment"])
        cameras = (
            list(corpus.fmb.episode(episode_id).cameras)
            if row["corpus_key"] == "fmb"
            else [camera["name"] for camera in record["cameras"]]
        )
        if row["corpus_key"] == "fmb":
            declared_dim = int(record["arrays"]["obs/q"]["shape"][1]) + 1
            anchor_mistake = _fmb_anchor_mistake(record, int(row["anchor_frame"]))
        else:
            declared_dim = int(record["action_dimension"])
            if declared_dim != int(record["state_dimension"]):
                raise ValueError(
                    f"{episode_id}: state width {record['state_dimension']} != action width {declared_dim}; "
                    "anchor encoding subtracts the state from the action and needs one width."
                )
            anchor_mistake = _common_anchor_mistake(record, float(row["anchor_s"]))
        if declared_dim != layout.dim:
            raise ValueError(
                f"{episode_id}: layout {layout.name} declares {layout.dim}D but the episode stores "
                f"{declared_dim}D. Register a new layout rather than widening an existing one."
            )
        if anchor_mistake is None:
            raise ValueError(f"{episode_id}: anchor {row['anchor_s']}s falls in no reviewed interval.")

        row["action_layout"] = layout.name
        row["action_layout_id"] = layout.index
        row["native_action_dim"] = layout.dim
        row["camera_roles"] = camera_roles_for(row["source"], cameras)
        row["has_depth"] = row["corpus_key"] == "fmb"
        row["mistake_flag_as_stored"] = bool(row["mistake"])
        row["mistake"] = bool(anchor_mistake)
        corrected += int(bool(anchor_mistake) != bool(row["mistake_flag_as_stored"]))
        prepared.append(row)

    selection = DiverseActorSelection(
        corpus=corpus,
        rows=prepared,
        episode_records=records,
        mistake_flags_corrected=corrected,
    )
    if verify_counts:
        assert_selection_counts(selection)
    return selection


def assert_selection_counts(selection: DiverseActorSelection) -> None:
    """Refuse to train on a corpus that is not the audited one."""
    episodes = len(selection.episode_ids)
    anchors = len(selection.rows)
    if episodes != EXPECTED_EPISODES or anchors != EXPECTED_ANCHORS:
        raise ValueError(
            f"diverse selection has {episodes} episodes / {anchors} anchors, expected "
            f"{EXPECTED_EPISODES} / {EXPECTED_ANCHORS}. The corpus on disk is not the one "
            "DIVERSE_ROBOT_DATASET.md describes -- rebuild the views or fix the roots."
        )
    per_source_episodes: dict[str, set[str]] = defaultdict(set)
    per_source_anchors: Counter[str] = Counter()
    for row in selection.rows:
        per_source_episodes[row["source"]].add(row["episode_id"])
        per_source_anchors[row["source"]] += 1
    for source, expected in EXPECTED_EPISODES_BY_SOURCE.items():
        actual = len(per_source_episodes.get(source, ()))
        if actual != expected:
            raise ValueError(f"source {source}: {actual} episodes, expected {expected}.")
    for source, expected in EXPECTED_ANCHORS_BY_SOURCE.items():
        actual = per_source_anchors[source]
        if actual != expected:
            raise ValueError(f"source {source}: {actual} anchors, expected {expected}.")


# ── Audit (gate A) ───────────────────────────────────────────────────────────


@dataclass
class SourceAudit:
    source: str
    episodes: int
    anchors: int
    embodiments: Counter = field(default_factory=Counter)
    action_layouts: Counter = field(default_factory=Counter)
    native_dims: Counter = field(default_factory=Counter)
    camera_role_sets: Counter = field(default_factory=Counter)
    resolutions: Counter = field(default_factory=Counter)
    native_rates: Counter = field(default_factory=Counter)
    splits: Counter = field(default_factory=Counter)
    quality_values: Counter = field(default_factory=Counter)
    quality_provenance: Counter = field(default_factory=Counter)
    mistake_anchors: int = 0
    mistake_anchors_as_stored: int = 0
    subtasks: int = 0
    depth_episodes: int = 0
    depth_anchors: int = 0
    future_inside_subtask: int = 0


def audit_selection(selection: DiverseActorSelection) -> dict[str, SourceAudit]:
    """Deterministic per-source counts: the numbers gate A compares to the ledger."""
    per_source: dict[str, SourceAudit] = {}
    episodes_seen: dict[str, set[str]] = defaultdict(set)
    subtasks: dict[str, set[str]] = defaultdict(set)

    for row in selection.rows:
        source = row["source"]
        audit = per_source.get(source)
        if audit is None:
            audit = per_source[source] = SourceAudit(source=source, episodes=0, anchors=0)
        audit.anchors += 1
        audit.embodiments[row["embodiment"]] += 1
        audit.action_layouts[row["action_layout"]] += 1
        audit.native_dims[row["native_action_dim"]] += 1
        audit.camera_role_sets["+".join(sorted(row["camera_roles"]))] += 1
        audit.native_rates[float(row["native_rate_hz"])] += 1
        audit.splits[row["split"]] += 1
        audit.quality_values[row["quality"]] += 1
        audit.quality_provenance[row.get("quality_provenance")] += 1
        audit.mistake_anchors += int(bool(row["mistake"]))
        audit.mistake_anchors_as_stored += int(bool(row["mistake_flag_as_stored"]))
        audit.depth_anchors += int(bool(row["has_depth"]))
        audit.future_inside_subtask += int(bool(row["future_inside_subtask"]))
        subtasks[source].add(row["subtask"])

        episode_id = row["episode_id"]
        if episode_id not in episodes_seen[source]:
            episodes_seen[source].add(episode_id)
            audit.episodes += 1
            audit.depth_episodes += int(bool(row["has_depth"]))
            record = selection.episode_records[episode_id]
            if row["corpus_key"] == "fmb":
                audit.resolutions["256x256"] += 1
            else:
                for size in sorted({f"{c['height']}x{c['width']}" for c in record["cameras"]}):
                    audit.resolutions[size] += 1

    for source, audit in per_source.items():
        audit.subtasks = len(subtasks[source])
    return per_source


def format_audit(selection: DiverseActorSelection) -> str:
    per_source = audit_selection(selection)
    lines: list[str] = []
    lines.append("Diverse actor selection -- actor_anchors(split=None, retained_only=True)")
    lines.append(f"  episodes {len(selection.episode_ids)} (expected {EXPECTED_EPISODES})")
    lines.append(f"  anchors  {len(selection.rows)} (expected {EXPECTED_ANCHORS})")
    lines.append(
        f"  mistake flags corrected from segment-level to anchor-level: "
        f"{selection.mistake_flags_corrected}"
    )
    lines.append("")
    header = f"{'source':<16}{'eps':>5}{'anchors':>9}{'dof':>5}{'cams':>6}{'depth eps':>11}{'subtasks':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for source in sorted(per_source, key=lambda name: -per_source[name].anchors):
        audit = per_source[source]
        dof = "/".join(str(dim) for dim in sorted(audit.native_dims))
        cams = "/".join(str(len(key.split("+"))) for key in sorted(audit.camera_role_sets))
        lines.append(
            f"{source:<16}{audit.episodes:>5}{audit.anchors:>9}{dof:>5}{cams:>6}"
            f"{audit.depth_episodes:>11}{audit.subtasks:>10}"
        )
    lines.append("")
    for source in sorted(per_source, key=lambda name: -per_source[name].anchors):
        audit = per_source[source]
        lines.append(f"[{source}]")
        lines.append(f"  layouts        {dict(audit.action_layouts)}")
        lines.append(f"  embodiments    {dict(audit.embodiments)}")
        lines.append(f"  camera roles   {dict(audit.camera_role_sets)}")
        lines.append(f"  resolutions    {dict(audit.resolutions)}")
        lines.append(f"  native rate Hz {dict(audit.native_rates)}")
        lines.append(f"  quality        {dict(sorted(audit.quality_values.items(), key=str))}")
        lines.append(f"  quality prov.  {dict(audit.quality_provenance)}")
        lines.append(
            f"  mistake anchor {audit.mistake_anchors} "
            f"(stored segment-level flag: {audit.mistake_anchors_as_stored})"
        )
        lines.append(f"  future inside  {audit.future_inside_subtask}/{audit.anchors}")
        lines.append(f"  depth anchors  {audit.depth_anchors}/{audit.anchors}")
        lines.append(f"  split (provenance only, never filtered) {dict(audit.splits)}")
    return "\n".join(lines)


# ── Sample contract (gate A) ─────────────────────────────────────────────────


def check_sample_contract(
    selection: DiverseActorSelection,
    *,
    per_source: int | None = None,
    history_ages_s: list[float] | None = None,
    cameras: bool = False,
) -> str:
    """Assert the loaded sample matches what training expects, on real samples.

    Seven packed observations; the configured history ages and the current frame all
    addressable inside them; a complete ``(30, native_D)`` future chunk at 30 Hz. With
    ``per_source=None`` every anchor is checked, grouped by episode so the reader's
    array cache is hit once per episode instead of once per anchor.
    """
    import numpy as np

    ages = history_ages_s if history_ages_s is not None else [6.0, 4.0, 2.0]
    slots = packed_history_slots(ages)

    by_episode = selection.rows_by_episode()
    order = sorted(by_episode)
    if per_source is not None:
        kept: dict[str, list[dict[str, Any]]] = {}
        taken: Counter[str] = Counter()
        for episode_id in order:
            rows = by_episode[episode_id]
            source = rows[0]["source"]
            if taken[source] >= per_source:
                continue
            take = rows[: max(1, per_source - taken[source])]
            taken[source] += len(take)
            kept[episode_id] = take
        by_episode = kept
        order = sorted(kept)

    checked = 0
    shapes: Counter[str] = Counter()
    for episode_id in order:
        for row in by_episode[episode_id]:
            sample = selection.corpus.actor_sample(row, cameras=cameras)
            state = np.asarray(sample["observation.state"])
            timestamps = np.asarray(sample["observation.timestamps"])
            action = np.asarray(sample["action"])
            native_dim = int(row["native_action_dim"])

            if state.shape != (PACKED_OBSERVATIONS, native_dim):
                raise ValueError(
                    f"{episode_id} anchor {row['anchor_s']}s: state {state.shape}, expected "
                    f"{(PACKED_OBSERVATIONS, native_dim)}."
                )
            if timestamps.shape != (PACKED_OBSERVATIONS,):
                raise ValueError(f"{episode_id}: {timestamps.shape[0]} packed observations, expected 7.")
            if action.shape != (FUTURE_POINTS, native_dim):
                raise ValueError(
                    f"{episode_id} anchor {row['anchor_s']}s: action {action.shape}, expected "
                    f"{(FUTURE_POINTS, native_dim)}."
                )
            if native_dim not in (7, 8):
                raise ValueError(f"{episode_id}: native width {native_dim} is neither 7 nor 8.")

            # The selected instants must be the ones the corpus actually packed. Clamped
            # history (an anchor closer to the episode start than 6 s) is not silently
            # tolerated: the anchor stride starts at 6 s precisely so it cannot happen.
            anchor_time = timestamps[PACKED_CURRENT_SLOT]
            for slot, age in zip(slots, ages, strict=True):
                error = abs((anchor_time - timestamps[slot]) - abs(age))
                if error > max(1.0 / float(row["native_rate_hz"]), 1e-6):
                    raise ValueError(
                        f"{episode_id} anchor {row['anchor_s']}s: slot {slot} sits "
                        f"{anchor_time - timestamps[slot]:.4f}s back, wanted {abs(age):.4f}s."
                    )
            action_times = np.asarray(sample["action.timestamps"])
            span = float(action_times[-1] - action_times[0])
            if abs(span - (FUTURE_POINTS - 1) / FUTURE_RATE_HZ) > 1e-6:
                raise ValueError(f"{episode_id}: future span {span:.6f}s is not 29/30 s at 30 Hz.")
            shapes[f"({FUTURE_POINTS}, {native_dim})"] += 1
            checked += 1

    scope = "all anchors" if per_source is None else f"<= {per_source} anchors/source"
    return (
        f"Sample contract OK on {checked} samples ({scope}): 7 packed observations, "
        f"history slots {slots} at ages {ages} s plus current slot {PACKED_CURRENT_SLOT}, "
        f"action shapes {dict(shapes)}."
    )


def open_federated_corpus(root: str | Path, *, actor_view: str = "actor_anchors_5hz.jsonl"):
    root = Path(root)
    return FederatedDiverseCorpus(root / "corpus", root / "fmb", actor_view=actor_view)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/diverse_robot_dataset")
    parser.add_argument(
        "--check-samples",
        type=int,
        default=0,
        help="Anchors per source to load for the sample-contract check; -1 checks every anchor.",
    )
    args = parser.parse_args()

    corpus = open_federated_corpus(args.root)
    selection = select_actor_anchors(corpus)
    print(format_audit(selection))

    if args.check_samples:
        print()
        per_source = None if args.check_samples < 0 else args.check_samples
        print(check_sample_contract(selection, per_source=per_source))


if __name__ == "__main__":
    main()
