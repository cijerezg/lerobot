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

"""Processor pipelines for tinypi05v2.

Processor logic is unchanged from `tinypi05` — the goal of v2 is purely to
eliminate the modeling-layer dependency on `transformers` / `pistar06` / `pi05`.
This wrapper simply forwards the v2 config to the existing
`make_tinypi05_pre_post_processors` factory.
"""

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.tinypi05.processor_tinypi05 import make_tinypi05_pre_post_processors
from lerobot.policies.tinypi05v2.configuration_tinypi05v2 import TinyPI05V2Config
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_tinypi05v2_pre_post_processors(
    config: TinyPI05V2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_tinypi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
