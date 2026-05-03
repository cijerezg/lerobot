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

"""Configuration for `tinypi05v2`.

`TinyPI05V2Config` is a drop-in subclass of `TinyPI05Config`. It re-registers
under the name `"tinypi05v2"` so the same checkpoint config can be loaded by
changing only the `type` string. All fields are identical; only the modeling
layer changes (see `modeling_tinypi05v2.py` for the self-contained rewrite that
drops the `transformers` / `pistar06` / `pi05` dependency chain).
"""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.tinypi05.configuration_tinypi05 import TinyPI05Config


@PreTrainedConfig.register_subclass("tinypi05v2")
@dataclass
class TinyPI05V2Config(TinyPI05Config):
    """Configuration for the self-contained tinypi05v2 policy.

    Same fields as `TinyPI05Config` — this subclass exists purely so the policy
    factory can route `"tinypi05v2"` checkpoints to the new single-file modeling
    implementation without touching the original `tinypi05` wiring.
    """
