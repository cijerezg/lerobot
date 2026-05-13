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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep  # noqa: F401
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_tinypi05 import TinyPI05Config, TinyPI05V2Config


@ProcessorStepRegistry.register(name="tinypi05_action_encoding_processor_step")
@dataclass
class TinyPI05ActionEncodingProcessorStep(ProcessorStep):
    """Encode chunk actions as absolute, anchor-relative, or delta-relative values."""

    action_encoding: str = "anchor"

    def __post_init__(self) -> None:
        if self.action_encoding not in ["absolute", "anchor", "delta"]:
            raise ValueError(
                f"action_encoding must be 'absolute', 'anchor', or 'delta' (got {self.action_encoding})"
            )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None or self.action_encoding == "absolute":
            return transition

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            return transition

        if not isinstance(action, torch.Tensor) or not isinstance(state, torch.Tensor):
            return transition

        state = state.to(device=action.device, dtype=action.dtype)
        if action.ndim == 3:
            state = state[:, None, :]
            action_state_dim = min(action.shape[-1], state.shape[-1])
            encoded = action.clone()
            if self.action_encoding == "anchor":
                encoded[..., :action_state_dim] = action[..., :action_state_dim] - state[..., :action_state_dim]
            else:
                delta = action.clone()
                delta[:, 0, :action_state_dim] = action[:, 0, :action_state_dim] - state[:, 0, :action_state_dim]
                delta[:, 1:, :action_state_dim] = (
                    action[:, 1:, :action_state_dim] - action[:, :-1, :action_state_dim]
                )
                encoded = delta
            transition[TransitionKey.ACTION] = encoded
            return transition

        action_state_dim = min(action.shape[-1], state.shape[-1])
        encoded = action.clone()
        encoded[..., :action_state_dim] = action[..., :action_state_dim] - state[..., :action_state_dim]
        transition[TransitionKey.ACTION] = encoded
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"action_encoding": self.action_encoding}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_tinypi05_pre_post_processors(
    config: TinyPI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        TinyPI05ActionEncodingProcessorStep(action_encoding=config.action_encoding),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name=config.tokenizer_name,
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def make_tinypi05v2_pre_post_processors(
    config: TinyPI05V2Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    return make_tinypi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
