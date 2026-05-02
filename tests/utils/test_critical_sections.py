#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import json
from pathlib import Path

import pytest
import torch

from lerobot.utils.critical_sections import (
    CriticalSection,
    CriticalSectionsProvider,
    parse_critical_sections,
)


def _make_doc(episodes: dict[str, dict]) -> dict:
    return {"episodes": episodes}


def test_parse_critical_sections_basic():
    raw = _make_doc(
        {
            "0": {"sections": [{"start": 1.0, "end": 2.0}]},
            "3": {"sections": [{"start": 0.5, "end": 0.7, "weight": 9.0}]},
            "7": {"sections": []},
        }
    )
    parsed = parse_critical_sections(raw, default_weight=5.0)

    assert parsed[0] == [CriticalSection(start=1.0, end=2.0, weight=5.0)]
    assert parsed[3] == [CriticalSection(start=0.5, end=0.7, weight=9.0)]
    assert parsed[7] == []


def test_parse_critical_sections_swaps_inverted_bounds(caplog):
    raw = _make_doc({"0": {"sections": [{"start": 2.0, "end": 1.0}]}})
    parsed = parse_critical_sections(raw, default_weight=5.0)
    assert parsed[0] == [CriticalSection(start=1.0, end=2.0, weight=5.0)]


def test_parse_critical_sections_rejects_missing_bounds():
    raw = _make_doc({"0": {"sections": [{"start": 1.0}]}})
    with pytest.raises(ValueError, match="missing `start`/`end`"):
        parse_critical_sections(raw)


def test_parse_critical_sections_rejects_non_int_episode():
    raw = _make_doc({"abc": {"sections": []}})
    with pytest.raises(ValueError, match="integer-valued"):
        parse_critical_sections(raw)


def test_provider_from_path(tmp_path: Path):
    doc = _make_doc(
        {
            "1": {"sections": [{"start": 0.0, "end": 0.5}]},
            "2": {"sections": []},
        }
    )
    path = tmp_path / "critical_sections.json"
    path.write_text(json.dumps(doc))

    provider = CriticalSectionsProvider.from_path(path, default_weight=4.0)
    assert provider.annotated_episode_indices() == [1]
    assert provider.num_annotated_episodes() == 1
    assert provider.sections_for_episode(1) == [
        CriticalSection(start=0.0, end=0.5, weight=4.0)
    ]


def test_compute_timestep_weights_basic():
    sections = {
        0: [CriticalSection(start=1.0, end=1.2, weight=5.0)],
        1: [],
    }
    provider = CriticalSectionsProvider(sections)

    fps = 10.0
    chunk_size = 4
    # Sample 0: ts=0.8 -> abs times {0.8, 0.9, 1.0, 1.1}.
    #   The first two are outside [1.0, 1.2]; 1.0 and 1.1 are inside.
    # Sample 1: episode 1 has no sections -> all weights are 1.0.
    batch = {
        "episode_index": torch.tensor([0, 1]),
        "timestamp": torch.tensor([0.8, 0.0]),
    }

    weights = provider.compute_timestep_weights(batch, chunk_size=chunk_size, fps=fps)

    assert weights.shape == (2, chunk_size)
    assert torch.allclose(weights[0], torch.tensor([1.0, 1.0, 5.0, 5.0]))
    assert torch.allclose(weights[1], torch.ones(chunk_size))


def test_compute_timestep_weights_overlap_takes_max():
    sections = {
        0: [
            CriticalSection(start=0.0, end=1.0, weight=2.0),
            CriticalSection(start=0.4, end=1.4, weight=7.0),
        ],
    }
    provider = CriticalSectionsProvider(sections)

    fps = 10.0
    batch = {
        "episode_index": torch.tensor([0]),
        "timestamp": torch.tensor([0.0]),
    }
    weights = provider.compute_timestep_weights(batch, chunk_size=20, fps=fps)
    # Steps every 0.1s: 0.0..1.9
    # [0.0, 1.0] alone (low side, before the heavier span kicks in): 0.0..0.3 -> 2.0
    # union with [0.4, 1.4]: 0.4..1.0 (overlap, max=7.0); 1.1..1.4 (only heavy span -> 7.0)
    # outside both: 1.5..1.9 -> 1.0
    expected = torch.tensor(
        [2.0] * 4 + [7.0] * 11 + [1.0] * 5
    )
    assert torch.allclose(weights[0], expected)


def test_compute_timestep_weights_empty_provider_is_all_ones():
    provider = CriticalSectionsProvider({})
    batch = {
        "episode_index": torch.tensor([0, 1, 2]),
        "timestamp": torch.tensor([0.1, 0.2, 0.3]),
    }
    weights = provider.compute_timestep_weights(batch, chunk_size=5, fps=10.0)
    assert torch.allclose(weights, torch.ones(3, 5))


def test_compute_timestep_weights_validates_inputs():
    provider = CriticalSectionsProvider({})
    batch = {
        "episode_index": torch.tensor([0]),
        "timestamp": torch.tensor([0.0]),
    }
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        provider.compute_timestep_weights(batch, chunk_size=0, fps=10.0)
    with pytest.raises(ValueError, match="fps must be positive"):
        provider.compute_timestep_weights(batch, chunk_size=2, fps=0.0)
    with pytest.raises(KeyError):
        provider.compute_timestep_weights(
            {"episode_index": torch.tensor([0])}, chunk_size=2, fps=10.0
        )
