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

import argparse
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.tinypi05.configuration_tinypi05 import TinyPI05Config
from lerobot.scripts.lerobot_train import train
from lerobot.utils.import_utils import register_third_party_plugins


DEFAULT_DATASET = "jackvial/cube-subtasks-e30-base120trim-0-9-101-end-fixed"
DEFAULT_DATASET_ROOT = "outputs/cube-subtasks-e30-base120trim-0-9-101-end-fixed"


def _optional_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _architecture_overrides(args: argparse.Namespace) -> dict:
    fields = [
        "vlm_width",
        "vlm_depth",
        "vlm_mlp_dim",
        "vlm_num_heads",
        "vlm_num_kv_heads",
        "vlm_head_dim",
        "expert_width",
        "expert_depth",
        "expert_mlp_dim",
        "expert_num_heads",
        "expert_num_kv_heads",
        "expert_head_dim",
        "vision_hidden_size",
        "vision_intermediate_size",
        "vision_num_hidden_layers",
        "vision_num_attention_heads",
        "vision_patch_size",
    ]
    return {field: getattr(args, field) for field in fields if getattr(args, field) is not None}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyPI05 from scratch on the cube subtasks dataset.")
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default="tinypi05_cube")
    parser.add_argument("--steps", type=_optional_positive_int, default=30_000)
    parser.add_argument("--batch-size", type=_optional_positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-freq", type=_optional_positive_int, default=5_000)
    parser.add_argument("--log-freq", type=_optional_positive_int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--architecture-preset", default="small_500m")
    parser.add_argument("--image-size", type=_optional_positive_int, default=224)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--optimizer-lr", type=float, default=2.5e-5)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1_000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30_000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")

    for field in [
        "vlm-width",
        "vlm-depth",
        "vlm-mlp-dim",
        "vlm-num-heads",
        "vlm-num-kv-heads",
        "vlm-head-dim",
        "expert-width",
        "expert-depth",
        "expert-mlp-dim",
        "expert-num-heads",
        "expert-num-kv-heads",
        "expert-head-dim",
        "vision-hidden-size",
        "vision-intermediate-size",
        "vision-num-hidden-layers",
        "vision-num-attention-heads",
        "vision-patch-size",
    ]:
        parser.add_argument(f"--{field}", type=_optional_positive_int, default=None)

    return parser.parse_args()


def main() -> None:
    register_third_party_plugins()
    args = parse_args()

    policy = TinyPI05Config(
        architecture_preset=args.architecture_preset,
        image_resolution=(args.image_size, args.image_size),
        dtype=args.dtype,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        freeze_vision_encoder=False,
        train_expert_only=False,
        optimizer_lr=args.optimizer_lr,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_decay_steps=args.scheduler_decay_steps,
        scheduler_decay_lr=args.scheduler_decay_lr,
        push_to_hub=False,
        **_architecture_overrides(args),
    )

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            use_imagenet_stats=False,
        ),
        policy=policy,
        output_dir=args.output_dir,
        job_name=args.job_name,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=WandBConfig(enable=args.wandb, project=args.wandb_project),
    )
    train(cfg)


if __name__ == "__main__":
    main()
