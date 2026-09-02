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

"""Canonical embodiment names for the prompt's embodiment clause.

Datasets spell their robot in whatever way their source did: DROID says
``franka_emika_panda_robotiq_2F-85``, RoboChallenge says ``ARX5`` or ``UR5``, the
ur7e corpus says ``ur7e``, the rebot roots say ``rebot_b601_follower``. The prompt
needs one short natural-language name per robot, and the replay buffer needs a
small integer to carry it — a buffer stores tensors, so the string itself cannot
ride along with the sample. This module owns both halves of that mapping.

``EMBODIMENT_NAMES`` is APPEND-ONLY. The index is what a trained checkpoint stores
in its processor config, so reordering or deleting an entry silently relabels every
robot in the prompt of every checkpoint trained before the change. New robots go on
the end.
"""

from __future__ import annotations

# Prompt-facing names, written the way the robots are named in the wild: these are
# proper nouns, and the tokenizer has seen "UR5" and "Franka" far more often than
# "ur5". Nothing downstream lowercases them (the prompt's only text normalization,
# _normalize_question_text, strips punctuation and prefixes but preserves case).
EMBODIMENT_NAMES: tuple[str, ...] = (
    "Franka Panda",
    "UR5",
    "UR7e",
    "ARX5",
    "Rebot B601",
    "SO-101",
    "ALOHA",
)

# The clause reads "The robot is {article} {name}." — spelled-out initialisms take
# "an" ("an ARX5", "an SO-101"), and "UR" is spoken "you-are" so it takes "a".
# Only exceptions are listed; the default is "a".
EMBODIMENT_ARTICLES: dict[str, str] = {
    "ARX5": "an",
    "SO-101": "an",
    "ALOHA": "an",
}

UNKNOWN_EMBODIMENT_INDEX = -1

# Raw ``robot_type`` spellings → canonical name. Keys are matched case-folded with
# separators stripped, so only genuinely different words need an entry here.
EMBODIMENT_ALIASES: dict[str, str] = {
    "franka": "Franka Panda",
    "frankapanda": "Franka Panda",
    "frankaemikapanda": "Franka Panda",
    "frankaemikapandarobotiq2f85": "Franka Panda",
    "panda": "Franka Panda",
    "fmb": "Franka Panda",
    "ur5": "UR5",
    "ur5e": "UR5",
    "ur7e": "UR7e",
    "arx5": "ARX5",
    "rebot": "Rebot B601",
    "rebotb601": "Rebot B601",
    "rebotb601follower": "Rebot B601",
    "rebotb601leader": "Rebot B601",
    "so101": "SO-101",
    "so101follower": "SO-101",
    "so101leader": "SO-101",
    "aloha": "ALOHA",
    "alohaagilex": "ALOHA",
}


def _alias_key(raw: str) -> str:
    """Case- and separator-insensitive key: ``franka_emika_panda_robotiq_2F-85`` →
    ``frankaemikapandarobotiq2f85``."""
    return "".join(character for character in raw.casefold() if character.isalnum())


def canonical_embodiment(raw: str | None) -> str | None:
    """Prompt name for a raw ``robot_type``, or None when it is unknown.

    Unknown is not an error: the clause is simply omitted, which is the same prompt
    the model saw before embodiment conditioning existed.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text in EMBODIMENT_NAMES:
        return text
    return EMBODIMENT_ALIASES.get(_alias_key(text))


def embodiment_index(raw: str | None) -> int:
    """Vocabulary index for a raw ``robot_type``, or -1 when it is unknown."""
    name = canonical_embodiment(raw)
    if name is None:
        return UNKNOWN_EMBODIMENT_INDEX
    return EMBODIMENT_NAMES.index(name)


def embodiment_article(name: str) -> str:
    """"a" or "an" for the embodiment clause."""
    return EMBODIMENT_ARTICLES.get(name, "a")


def embodiment_name(index: int, names: list[str] | tuple[str, ...] | None = None) -> str | None:
    """Prompt name for a vocabulary index, or None when it is unknown/out of range.

    ``names`` lets a checkpoint render through the vocabulary it was trained with
    rather than whatever this module currently declares.
    """
    vocabulary = EMBODIMENT_NAMES if names is None else names
    if index < 0 or index >= len(vocabulary):
        return None
    return vocabulary[index]
