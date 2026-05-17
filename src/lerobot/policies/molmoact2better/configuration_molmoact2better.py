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

"""Configuration for the `molmoact2better` policy variant.

`MolmoAct2BetterConfig` intentionally keeps the same fields as `MolmoAct2Config`.
The separate registered type lets the factory route the same checkpoints through
the optimized inference wrapper without changing saved MolmoAct2 checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config


@PreTrainedConfig.register_subclass("molmoact2better")
@dataclass
class MolmoAct2BetterConfig(MolmoAct2Config):
    """Drop-in MolmoAct2 config registered under `molmoact2better`."""
