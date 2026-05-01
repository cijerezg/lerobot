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

"""
Diagnostic probe scripts for trained policies.

Each probe is runnable both as a standalone CLI (``python -m lerobot.probes.<name>``)
and importable from the offline validation pipeline.

Available probes (import directly)::

    from lerobot.probes.offline_inference_pi05 import ...
    from lerobot.probes.attention_pi05 import ...
    from lerobot.probes.attention_spatial_memorization import ...
    from lerobot.probes.action_drift_jacobian import ...
    from lerobot.probes.actions_pi05 import ...
    from lerobot.probes.representations_pi05 import ...
    from lerobot.probes.utils_pi05 import ...
"""

__all__: list[str] = []
