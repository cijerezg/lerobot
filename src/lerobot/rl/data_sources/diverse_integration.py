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

"""Wire the diverse corpus into rl_offline.py alongside the ReBot buffers (plan phase J).

The two halves reach the training loop as different kinds of object -- a cached
``ReplayBuffer`` over a LeRobotDataset, and an anchor-indexed ``DiverseActorBuffer`` over
the source-native corpus -- and the loop must not care which is which. This module is the
seam: it builds the diverse half, brings the ReBot half onto the shared camera roles, and
returns the two mixture groups the hierarchical iterator consumes.

Everything it refuses to do is deliberate. It will not start on the wrong corpus (the
selection asserts the ledger), it will not fall back to a 28-hour video decode when a
cache is required, and it will not build a held-out diverse validation loader -- this run
trains on every accepted episode by decision, and a split-shaped object lying around is
how that decision gets quietly reversed.
"""

from __future__ import annotations

import logging
from typing import Any

from lerobot.configs.diverse import DiverseCollectionConfig
from lerobot.datasets.diverse_actor_selection import (
    CANONICAL_CAMERA_ROLES,
    MAX_ACTION_DIM,
    action_layout_by_name,
    open_federated_corpus,
    select_actor_anchors,
)
from lerobot.datasets.diverse_prompt import extend_dataset_vocabulary
from lerobot.rl.data_sources.diverse_actor_buffer import (
    DiverseActorBuffer,
    DiverseSampleSpec,
)
from lerobot.rl.data_sources.diverse_actor_cache import resolve_cache
from lerobot.rl.data_sources.diverse_mixture import HierarchicalAnchorSampler, MixtureGroup
from lerobot.rl.data_sources.rebot_role_adapter import (
    RoleAlignedBuffer,
    probe_rebot_cache,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


def _policy_image_roles(cfg) -> tuple[str, ...]:
    keys = [key for key in (cfg.policy.input_features or {}) if key.startswith(f"{OBS_IMAGES}.")]
    roles = tuple(key.rsplit(".", 1)[-1] for key in sorted(keys))
    unknown = [role for role in roles if role not in CANONICAL_CAMERA_ROLES]
    if unknown:
        raise ValueError(
            f"policy.input_features declares camera roles {unknown}, which are not canonical "
            f"({', '.join(CANONICAL_CAMERA_ROLES)}). A mixed run needs both halves on one "
            "camera vocabulary."
        )
    return roles


def _image_size(cfg) -> tuple[int, int]:
    feature = (cfg.policy.input_features or {}).get(f"{OBS_IMAGES}.{CANONICAL_CAMERA_ROLES[0]}")
    if feature is None:
        for key, value in (cfg.policy.input_features or {}).items():
            if key.startswith(f"{OBS_IMAGES}."):
                feature = value
                break
    if feature is None or not getattr(feature, "shape", None):
        raise ValueError("policy.input_features declares no image feature with a shape.")
    return int(feature.shape[1]), int(feature.shape[2])


def sample_spec_from_config(cfg) -> DiverseSampleSpec:
    """The sample contract this run's policy config implies."""
    memory_cfg = getattr(cfg.policy, "memory", None)
    ages = tuple(memory_cfg.history_times_seconds()) if memory_cfg is not None else (6.0, 4.0, 2.0)
    pointmap = getattr(cfg.policy, "pointmap_config", None)
    depth_role = pointmap.depth_key if pointmap is not None else CANONICAL_CAMERA_ROLES[-1]
    depth_size = tuple(pointmap.image_size) if pointmap is not None else _image_size(cfg)
    state_feature = (cfg.policy.input_features or {}).get(OBS_STATE)
    width = int(state_feature.shape[0]) if state_feature is not None else MAX_ACTION_DIM
    return DiverseSampleSpec(
        history_ages_s=ages,
        action_horizon=int(cfg.policy.chunk_size),
        max_width=width,
        camera_roles=_policy_image_roles(cfg),
        image_size=_image_size(cfg),
        depth_size=(int(depth_size[0]), int(depth_size[1])),
        depth_role=depth_role,
        load_images=True,
        load_depth=pointmap is not None,
    )


def build_diverse_buffer(
    cfg,
    diverse_cfg: DiverseCollectionConfig,
    *,
    main_dataset,
    device: str,
    seed: int,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
) -> DiverseActorBuffer:
    """The diverse half, cache-backed, sampled hierarchically, on the shared vocabulary."""
    diverse_cfg.validate()
    spec = sample_spec_from_config(cfg)
    selection = select_actor_anchors(open_federated_corpus(diverse_cfg.root))
    if is_main_process:
        logger.info(
            "[Diverse] selection: %d episodes, %d anchors (%d mistake flags corrected)",
            len(selection.episode_ids),
            len(selection.rows),
            selection.mistake_flags_corrected,
        )

    cache = resolve_cache(
        diverse_cfg.root,
        diverse_cfg.cache_dir or getattr(cfg, "buffer_cache_dir", None),
        spec,
        selection,
        cache_policy=str(getattr(cfg, "cache_policy", "fallback")),
    )
    task_indices, subtask_indices = extend_dataset_vocabulary(
        main_dataset, selection, is_main_process=is_main_process
    )
    sampler = HierarchicalAnchorSampler(
        selection,
        group_weight=diverse_cfg.group_weight,
        group_weight_overrides=diverse_cfg.group_weight_overrides,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    return DiverseActorBuffer(
        selection,
        spec,
        device=device,
        cache=cache,
        sampler=sampler,
        task_indices=task_indices,
        subtask_indices=subtask_indices,
        render_automatic_quality=diverse_cfg.render_automatic_quality,
    )


def align_rebot_buffers(
    rebot_buffers: list[Any],
    diverse_cfg: DiverseCollectionConfig,
    spec: DiverseSampleSpec,
    *,
    is_main_process: bool = True,
) -> list[RoleAlignedBuffer]:
    """Bring each ReBot buffer onto the canonical roles and identity columns."""
    layout = action_layout_by_name(diverse_cfg.rebot_layout)
    aligned = []
    for buffer in rebot_buffers:
        adapter = RoleAlignedBuffer(
            buffer,
            camera_roles=spec.camera_roles,
            action_layout_id=layout.index,
            image_size=spec.image_size,
            depth_role=spec.depth_role,
            depth_intrinsics=diverse_cfg.rebot_depth_intrinsics,
        )
        if is_main_process:
            logger.info(
                "[Diverse] ReBot buffer aligned: present %s, absent %s",
                adapter.present_roles,
                adapter.absent_roles,
            )
        aligned.append(adapter)
    return aligned


def probe_rebot_caches(cache_dirs: list[Any], history_offsets_frames: list[int], depth_role: str) -> None:
    """Fail before training if a ReBot cache cannot serve this run's observations.

    ``depth_role`` is this run's canonical role; the columns on disk carry ReBot's own
    camera name, because the rename happens after loading. Ask under the name the cache
    was written with, or a healthy cache looks broken.
    """
    from lerobot.datasets.diverse_actor_selection import CAMERA_ROLE_MAP

    cache_camera = next(
        (camera for camera, role in CAMERA_ROLE_MAP["rebot"].items() if role == depth_role), None
    )
    problems: list[str] = []
    for cache_dir in cache_dirs:
        report = probe_rebot_cache(
            cache_dir, history_offsets_frames=history_offsets_frames, depth_role=cache_camera
        )
        logger.info("[Diverse] %s", report.describe())
        if not report.usable:
            problems.append(f"{cache_dir}: {'; '.join(report.problems)}")
    if problems:
        raise RuntimeError(
            "ReBot replay caches cannot serve this run's observation contract:\n  "
            + "\n  ".join(problems)
            + "\nRebuild them with lerobot_memmap_buffer_cache.py at the required stride."
        )


def build_mixture_groups(
    diverse_cfg: DiverseCollectionConfig,
    rebot_buffers: list[Any],
    diverse_buffer: DiverseActorBuffer,
    *,
    rebot_weights: list[float] | None = None,
) -> list[MixtureGroup]:
    """ReBot first, diverse second -- the order that decides odd-batch ties."""
    return [
        MixtureGroup(
            "rebot",
            list(rebot_buffers),
            weight=diverse_cfg.rebot_weight,
            inner_weights=list(rebot_weights) if rebot_weights else None,
        ),
        MixtureGroup("diverse", [diverse_buffer], weight=diverse_cfg.weight),
    ]
