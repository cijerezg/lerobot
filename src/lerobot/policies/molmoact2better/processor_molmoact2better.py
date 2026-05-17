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

from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.molmoact2.processor_molmoact2 import make_molmoact2_pre_post_processors
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_molmoact2better import MolmoAct2BetterConfig


def make_molmoact2better_pre_post_processors(
    config: MolmoAct2BetterConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    dataset_meta: Any | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_molmoact2_pre_post_processors(
        config=config,
        dataset_stats=dataset_stats,
        dataset_meta=dataset_meta,
    )
