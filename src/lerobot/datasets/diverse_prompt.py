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

"""Prompt vocabularies and label provenance for the diverse corpus (plan phase G).

The corpus keeps its annotations source-native, and the prompt pipeline expects small
integers plus a pinned vocabulary. This module is the join: it reads the corpus records
directly (no LeRobot sidecars), folds the diverse task and subtask strings into the
master ReBot vocabulary by the same conservative text matching the collection loader
already uses for its peers, and turns the label provenance into stable ids a batch can
carry.

The provenance is the point. DIVERSE_ROBOT_DATASET.md section 6 is explicit that quality
means different things per source: DROID and UR7e are human-reviewed ordinals, FMB is
human-reviewed against the ReBot rubric, and RoboChallenge was originally derived automatically. Its 334 retained segments have now
been visually reviewed against the ReBot rubric by Codex, independently confirming quality 5
and no semantic mistake, with ``model_reviewed_rebot_rubric`` provenance. This remains distinct
from a human review and is therefore not rendered in prompts by default.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.datasets.diverse_actor_selection import DiverseActorSelection

logger = logging.getLogger(__name__)

# APPEND-ONLY. A batch carries the id, so reordering relabels every past run.
QUALITY_PROVENANCE: tuple[str, ...] = (
    "unknown",
    "human_reviewed",
    "human_reviewed_rebot_rubric",
    "source_derived_automatic",
    "model_reviewed_rebot_rubric",
)

# Provenances where a person actually looked. Only these earn a rendered quality clause
# by default; see `should_render_quality`.
HUMAN_REVIEWED_PROVENANCE: frozenset[str] = frozenset(
    {"human_reviewed", "human_reviewed_rebot_rubric"}
)

RETENTION_REASONS: tuple[str, ...] = (
    "unknown",
    "useful_motion",
    "informative_mistake",
    "recovery",
)

UNKNOWN_QUALITY = -1.0


def quality_provenance_id(name: str | None) -> int:
    if name is None:
        return 0
    try:
        return QUALITY_PROVENANCE.index(str(name))
    except ValueError as error:
        raise KeyError(
            f"unknown quality provenance {name!r}. Append it to QUALITY_PROVENANCE -- never "
            "reuse an existing id, and never fold a new provenance into an old one."
        ) from error


def retention_reason_id(name: str | None) -> int:
    if name is None:
        return 0
    try:
        return RETENTION_REASONS.index(str(name))
    except ValueError as error:
        raise KeyError(f"unknown retention reason {name!r}. Append it to RETENTION_REASONS.") from error


def should_render_quality(provenance: str | None, *, render_automatic: bool = False) -> bool:
    """Whether this sample's quality integer may reach the prompt."""
    if render_automatic:
        return provenance is not None
    return str(provenance) in HUMAN_REVIEWED_PROVENANCE


# ── Episode task text ────────────────────────────────────────────────────────

# FMB stores no episode-level task string. Every one of its 100 accepted episodes ends
# in the `insert` primitive (asserted in the corpus regression tests), so the episode
# task is read off the data rather than invented; the provenance below records that.
FMB_TASK_TEXT = "insert the object into the board"
FMB_TASK_PROVENANCE = "derived_from_terminal_primitive"


def episode_task(record: dict[str, Any], source: str) -> tuple[str, str]:
    """(task text, provenance) for one episode."""
    if source == "fmb":
        intervals = record.get("primitive_intervals") or []
        if not intervals or str(intervals[-1].get("primitive")) != "insert":
            raise ValueError(
                f"FMB episode {record.get('episode_id')!r} does not end in `insert`; the derived "
                "episode task no longer describes it. Read the task from the source instead of "
                "extending this rule."
            )
        return FMB_TASK_TEXT, FMB_TASK_PROVENANCE
    task = record.get("task")
    if not task:
        raise ValueError(
            f"episode {record.get('episode_id')!r} of source {source!r} has no task string, and "
            "there is no rule for deriving one. Omitting the task would leave the prompt's only "
            "goal clause empty."
        )
    return str(task), "source_recorded"


# ── Vocabulary ───────────────────────────────────────────────────────────────


def diverse_vocabulary(selection: DiverseActorSelection) -> tuple[list[str], list[str]]:
    """(tasks, subtasks) present in the training selection, deterministically ordered."""
    tasks: set[str] = set()
    subtasks: set[str] = set()
    for row in selection.rows:
        record = selection.episode_records[str(row["episode_id"])]
        tasks.add(episode_task(record, row["source"])[0])
        subtasks.add(str(row["subtask"]))
    return sorted(tasks), sorted(subtasks)


def extend_dataset_vocabulary(
    target_dataset,
    selection: DiverseActorSelection,
    *,
    is_main_process: bool = True,
) -> tuple[dict[str, int], dict[str, int]]:
    """Fold the diverse strings into the run's master vocabulary tables.

    ``target_dataset`` is the normalization-source ReBot dataset whose ``meta.tasks`` and
    ``meta.subtasks`` tables the collection loader already treats as the master -- the
    same tables ``remap_tasks_for_dataset`` appends peer roots into. Reusing them means
    the two halves of the mixture share one index space and one pinned prompt vocabulary,
    rather than a parallel one that could drift.

    Returns (task string -> index, subtask string -> index) for the diverse rows.
    """
    import pandas as pd

    from lerobot.rl.offline_dataset_utils import (
        _idx_to_subtask_name,
        _idx_to_task_name,
        _label_key,
    )

    tasks, subtasks = diverse_vocabulary(selection)
    result: list[dict[str, int]] = []
    for names, table_name, index_column, text_column, existing in (
        (tasks, "tasks", "task_index", "task", _idx_to_task_name(target_dataset)),
        (subtasks, "subtasks", "subtask_index", "subtask", _idx_to_subtask_name(target_dataset)),
    ):
        key_to_index = {_label_key(name): index for index, name in existing.items()}
        next_index = max(existing, default=-1) + 1
        mapping: dict[str, int] = {}
        additions: list[tuple[str, int]] = []
        for name in names:
            key = _label_key(name)
            if key in key_to_index:
                mapping[name] = key_to_index[key]
            else:
                mapping[name] = next_index
                key_to_index[key] = next_index
                additions.append((name, next_index))
                next_index += 1
        if additions:
            meta = target_dataset.meta
            table = getattr(meta, table_name, None)
            frame = pd.DataFrame(
                {index_column: [index for _, index in additions]},
                index=pd.Index([name for name, _ in additions], name=text_column),
            )
            setattr(meta, table_name, frame if table is None else pd.concat([table, frame]))
        if is_main_process:
            logger.info(
                "[DiversePrompt] %s: %d diverse entries, %d new (vocabulary now %d)",
                table_name,
                len(names),
                len(additions),
                next_index,
            )
        result.append(mapping)
    return result[0], result[1]
