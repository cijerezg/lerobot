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

"""CPU sample test for the mixed ReBot + diverse collection (integration plan gate J).

Builds exactly what rl_offline.py builds -- the ReBot buffers, the diverse buffer, the
hierarchical iterator and the real preprocessor -- draws batches, and reports what came
out. No model, no GPU, no gradient: this is the cheap check that runs before the smoke
training run and answers the questions that do not need a forward pass.

    uv run --no-project --python .venv/bin/python \\
        python -m lerobot.scripts.diverse_smoke --config_path=config_rl.yaml --batches 8
"""

from __future__ import annotations

import argparse
import logging

import torch

from lerobot.rl.data_sources.diverse_integration import (
    align_rebot_buffers,
    build_diverse_buffer,
    build_mixture_groups,
    probe_rebot_caches,
    sample_spec_from_config,
)
from lerobot.rl.data_sources.diverse_mixture import (
    MixtureTelemetry,
    allocate_group_quotas,
    make_hierarchical_offline_iterator,
)
from lerobot.utils.constants import ACTION, OBS_STATE

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

SOURCE_NAMES = {0: "droid", 1: "droid_success", 2: "fmb", 3: "robochallenge", 4: "ur7e", 5: "rebot"}


def _load_config(path: str):
    # Import for their registry side effects, the way the training entry point does.
    import draccus

    import lerobot.teleoperators.rebot_102_leader  # noqa: F401
    from lerobot.cameras import opencv, realsense  # noqa: F401
    from lerobot.configs.train import TrainRLServerPipelineConfig
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLConfig  # noqa: F401
    from lerobot.robots.rebot_b601_follower import RebotB601Follower  # noqa: F401

    return draccus.parse(TrainRLServerPipelineConfig, args=[f"--config_path={path}"])


def _rebot_buffers(cfg, history_offsets):
    from lerobot.rl.buffer import ReplayBuffer
    from lerobot.rl.offline_dataset_utils import (
        buffer_state_keys,
        get_offline_dataset_sources,
        load_metadata_rows,
        load_offline_dataset,
        materialize_dataset_labels,
    )

    sources = get_offline_dataset_sources(cfg)
    datasets, buffers = [], []
    for index, source in enumerate(sources):
        dataset = load_offline_dataset(cfg, source)
        cached = ReplayBuffer.find_cache(
            dataset,
            cfg.buffer_cache_dir,
            state_keys=buffer_state_keys(cfg),
            image_storage_dtype=cfg.policy.image_storage_dtype,
            image_storage_size=cfg.policy.image_storage_size,
            image_stride=cfg.policy.image_stride,
        )
        if cached is None:
            raise FileNotFoundError(f"no ReBot cache for {source.name!r} under {cfg.buffer_cache_dir!r}")
        buffer = ReplayBuffer.from_cache(
            cached, device="cpu", use_drq=False, history_offsets=history_offsets
        )
        materialize_dataset_labels(
            buffer,
            dataset,
            datasets[0] if datasets else dataset,
            source_index=index,
            is_main_process=True,
            require_depth_gripper_event_labels=bool(cfg.policy.depth_gripper_event_loss.enabled),
            embodiment=source.embodiment,
        )
        if cfg.policy.memory.metadata_enabled:
            buffer.materialize_metadata(*load_metadata_rows(dataset.root))
        datasets.append(dataset)
        buffers.append(buffer)
    return datasets[0], buffers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="config_rl.yaml")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--skip-preprocessor", action="store_true")
    parser.add_argument(
        "--cache-policy",
        default=None,
        choices=("require", "fallback"),
        help="Override the config, e.g. to smoke the uncached path before the cache exists.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config_path)
    if args.cache_policy is not None:
        cfg.cache_policy = args.cache_policy
    if not cfg.diverse.enabled:
        raise SystemExit(f"{args.config_path} has diverse.enabled=false; nothing to smoke.")
    spec = sample_spec_from_config(cfg)
    fps = float(getattr(cfg, "fps", 30) or 30)
    history_offsets = cfg.policy.memory.history_offsets(fps)
    logger.info("Sample contract: %s", spec.fingerprint())

    main_dataset, rebot = _rebot_buffers(cfg, history_offsets)
    if cfg.diverse.probe_rebot_caches:
        probe_rebot_caches(
            [buffer.cache_dir for buffer in rebot],
            history_offsets_frames=cfg.policy.memory.history_offsets_frames(fps),
            depth_role=spec.depth_role,
        )
    diverse = build_diverse_buffer(
        cfg, cfg.diverse, main_dataset=main_dataset, device="cpu", seed=int(cfg.seed or 0)
    )
    groups = build_mixture_groups(
        cfg.diverse,
        align_rebot_buffers(rebot, cfg.diverse, spec),
        diverse,
        rebot_weights=[source.weight for source in __import__(
            "lerobot.rl.offline_dataset_utils", fromlist=["get_offline_dataset_sources"]
        ).get_offline_dataset_sources(cfg)],
    )
    logger.info("Batch allocation: %s", allocate_group_quotas(cfg.batch_size, groups))

    preprocessor = None
    if not args.skip_preprocessor:
        from lerobot.rl.rl_trainer import Trainer

        preprocessor, _ = Trainer.for_config(cfg).make_processors(cfg, dataset=main_dataset)

    telemetry = MixtureTelemetry()
    iterator = make_hierarchical_offline_iterator(
        groups,
        batch_size=cfg.batch_size,
        async_prefetch=False,
        action_chunk_size=cfg.policy.n_action_steps,
    )
    for step in range(args.batches):
        batch = next(iterator)
        telemetry.observe(batch, depth_key=f"depth.{spec.depth_role}.depth")
        info = batch["complementary_info"]
        widths = (~info["action_dim_is_pad"]).sum(dim=1)
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (batch[ACTION], batch["state"][OBS_STATE])
        )
        logger.info(
            "batch %d: action %s widths %s finite=%s cameras/sample %s depth %d/%d",
            step,
            tuple(batch[ACTION].shape),
            sorted(set(widths.tolist())),
            finite,
            sorted(set(info["camera_is_present"].sum(dim=1).tolist())),
            int(info[f"depth.{spec.depth_role}.depth_is_present"].sum()),
            int(batch[ACTION].shape[0]),
        )
        if preprocessor is not None and step == 0:
            from lerobot.rl.rl_trainer import Trainer  # noqa: F401 (kept local)

            trainer = Trainer.for_config(cfg)
            observations = trainer._inject_depth_observations(
                dict(batch["state"]), info, cfg
            )
            packed = trainer.build_training_batch(
                raw_batch=batch,
                observations=observations,
                actions=batch[ACTION],
                preprocessor=preprocessor,
                dataset=main_dataset,
                cfg=cfg,
            )
            logger.info(
                "  preprocessed: input_ids %s attention %s labels %s",
                tuple(packed["input_ids"].shape),
                tuple(packed["attention_mask"].shape),
                tuple(packed["labels"].shape) if "labels" in packed else None,
            )
            attended = packed["attention_mask"].sum(dim=1)
            logger.info("  attended tokens per sample: min %d max %d", int(attended.min()), int(attended.max()))

    logger.info("\n%s", telemetry.describe(SOURCE_NAMES))


if __name__ == "__main__":
    main()
