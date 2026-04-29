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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@dataclass(frozen=True)
class TinyPI05Architecture:
    vlm_width: int
    vlm_depth: int
    vlm_mlp_dim: int
    vlm_num_heads: int
    vlm_num_kv_heads: int
    vlm_head_dim: int
    expert_width: int
    expert_depth: int
    expert_mlp_dim: int
    expert_num_heads: int
    expert_num_kv_heads: int
    expert_head_dim: int
    vision_hidden_size: int
    vision_intermediate_size: int
    vision_num_hidden_layers: int
    vision_num_attention_heads: int
    vision_patch_size: int


TINYPi05_ARCHITECTURE_PRESETS: dict[str, TinyPI05Architecture] = {
    "debug": TinyPI05Architecture(
        vlm_width=128,
        vlm_depth=2,
        vlm_mlp_dim=512,
        vlm_num_heads=8,
        vlm_num_kv_heads=1,
        vlm_head_dim=16,
        expert_width=128,
        expert_depth=2,
        expert_mlp_dim=512,
        expert_num_heads=8,
        expert_num_kv_heads=1,
        expert_head_dim=16,
        vision_hidden_size=128,
        vision_intermediate_size=512,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=4,
        vision_patch_size=16,
    ),
    "tiny_300m": TinyPI05Architecture(
        vlm_width=512,
        vlm_depth=8,
        vlm_mlp_dim=2048,
        vlm_num_heads=8,
        vlm_num_kv_heads=1,
        vlm_head_dim=64,
        expert_width=512,
        expert_depth=8,
        expert_mlp_dim=2048,
        expert_num_heads=8,
        expert_num_kv_heads=1,
        expert_head_dim=64,
        vision_hidden_size=512,
        vision_intermediate_size=2048,
        vision_num_hidden_layers=8,
        vision_num_attention_heads=8,
        vision_patch_size=16,
    ),
    "small_500m": TinyPI05Architecture(
        vlm_width=768,
        vlm_depth=10,
        vlm_mlp_dim=3072,
        vlm_num_heads=8,
        vlm_num_kv_heads=1,
        vlm_head_dim=96,
        expert_width=768,
        expert_depth=10,
        expert_mlp_dim=3072,
        expert_num_heads=8,
        expert_num_kv_heads=1,
        expert_head_dim=96,
        vision_hidden_size=768,
        vision_intermediate_size=3072,
        vision_num_hidden_layers=10,
        vision_num_attention_heads=12,
        vision_patch_size=16,
    ),
}


@PreTrainedConfig.register_subclass("tinypi05")
@dataclass
class TinyPI05Config(PreTrainedConfig):
    """Configuration for a scaled-down PI0.5-style policy.

    `tinypi05` keeps the PI0.5 PaliGemma-prefix plus Gemma-action-expert
    architecture but lets the model dimensions be set from scratch.
    """

    architecture_preset: str = "small_500m"

    # Architecture overrides. Leave as None to use `architecture_preset`.
    vlm_width: int | None = None
    vlm_depth: int | None = None
    vlm_mlp_dim: int | None = None
    vlm_num_heads: int | None = None
    vlm_num_kv_heads: int | None = None
    vlm_head_dim: int | None = None
    expert_width: int | None = None
    expert_depth: int | None = None
    expert_mlp_dim: int | None = None
    expert_num_heads: int | None = None
    expert_num_kv_heads: int | None = None
    expert_head_dim: int | None = None
    vision_hidden_size: int | None = None
    vision_intermediate_size: int | None = None
    vision_num_hidden_layers: int | None = None
    vision_num_attention_heads: int | None = None
    vision_patch_size: int | None = None

    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 8
    time_sampling_beta_alpha: float = 1.0
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    empty_cameras: int = 0
    tokenizer_max_length: int = 200
    tokenizer_name: str = "google/paligemma-3b-pt-224"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    gradient_checkpointing: bool = True
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        if self.architecture_preset not in TINYPi05_ARCHITECTURE_PRESETS:
            valid = ", ".join(sorted(TINYPi05_ARCHITECTURE_PRESETS))
            raise ValueError(f"Invalid architecture_preset={self.architecture_preset!r}. Valid presets: {valid}")

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        arch = self.resolved_architecture()
        if arch.vlm_depth != arch.expert_depth:
            raise ValueError(
                "tinypi05 currently requires vlm_depth == expert_depth because it uses PI0.5-style "
                "paired prefix/expert attention layers."
            )
        if arch.vlm_num_heads != 8 or arch.expert_num_heads != 8:
            raise ValueError(
                "tinypi05 currently requires 8 VLM and expert attention heads because the reused "
                "PI0.5 joint-attention kernel assumes 8 heads."
            )
        if self.image_resolution[0] != self.image_resolution[1]:
            raise ValueError(f"PaliGemma expects square image resolution, got {self.image_resolution}")

    def resolved_architecture(self) -> TinyPI05Architecture:
        preset = TINYPi05_ARCHITECTURE_PRESETS[self.architecture_preset]
        values = {
            field_name: (
                getattr(self, field_name)
                if getattr(self, field_name) is not None
                else getattr(preset, field_name)
            )
            for field_name in preset.__dataclass_fields__
        }
        return TinyPI05Architecture(**values)

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
