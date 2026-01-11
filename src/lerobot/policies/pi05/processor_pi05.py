#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import pad_vector
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
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    advantage_key: str = "advantage"
    advantage_scaling: float = 1.0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")
        
        # Advantage is optional (default to 1.0 if not present, or use inference_advantage)
        advantages = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.advantage_key)
        if advantages is None:
            # Fallback to a default value if not provided (e.g. during inference)
            # We assume it's a tensor of shape [B, 1] or [B]
            batch_size = len(tasks)
            advantages = torch.ones((batch_size, 1), dtype=torch.float32)

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.to(dtype=torch.float32).cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Process Advantage with 5 bins
        if isinstance(advantages, torch.Tensor):
            adv_tensor = advantages.to(dtype=torch.float32)
        else:
            adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        
        if adv_tensor.dim() == 1:
            adv_tensor = adv_tensor.unsqueeze(-1)
            
        # Squash advantage to [-1, 1] using tanh after scaling
        squashed_adv = torch.tanh(adv_tensor / self.advantage_scaling)
        adv_np = squashed_adv.cpu().numpy()

        # Use user-defined bins: [-1.0, -0.8, -0.4, 0.4, 0.8, 1.0]
        # Bins:
        # 0: < -0.8 (Very Negative)
        # 1: -0.8 to -0.4 (Negative)
        # 2: -0.4 to 0.4 (Neutral)
        # 3: 0.4 to 0.8 (Positive)
        # 4: > 0.8 (Very Positive)
        
        # Note: np.digitize returns 1-based indices for bins.
        # bins=[-0.8, -0.4, 0.4, 0.8]
        # x < -0.8 -> 0
        # -0.8 <= x < -0.4 -> 1
        # -0.4 <= x < 0.4 -> 2
        # 0.4 <= x < 0.8 -> 3
        # x >= 0.8 -> 4
        bins = np.array([-0.8, -0.4, 0.4, 0.8])
        bin_indices = np.digitize(adv_np, bins)
        
        labels = ["very negative", "negative", "neutral", "positive", "very positive"]
        discretized_adv_labels = [labels[idx] for idx in bin_indices.flatten()]

        full_prompts = []
        critic_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            adv_str = discretized_adv_labels[i]
            
            # Actor prompt (with advantage)
            full_prompt = f"Task: {cleaned_text}, State: {state_str}, Advantage: {adv_str};\nAction: "
            full_prompts.append(full_prompt)
            
            # Critic prompt (WITHOUT advantage)
            # Format: "Task: {task}, State: {state}"
            critic_prompt = f"Task: {cleaned_text}, State: {state_str}"
            critic_prompts.append(critic_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        transition[TransitionKey.COMPLEMENTARY_DATA]["critic_prompt"] = critic_prompts
        
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@dataclass
class CriticTokenizerProcessorStep(TokenizerProcessorStep):
    """
    Processor step to tokenize the critic prompt (without advantage).
    """
    task_key: str = "critic_prompt"
    
    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Critic prompt cannot be None")

        # Tokenize the task
        tokenized_prompt = self._tokenize_text(task)

        # Detect device
        target_device = self._detect_device(self.transition)

        # Move to device
        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }

        # Create new observation dict
        new_observation = dict(observation)

        # Add tokenized data with CRITIC specific keys
        # We use "critic_tokens" and "critic_pad_mask" as expected by PI05RLPolicy
        new_observation["critic_tokens"] = tokenized_prompt["input_ids"]
        new_observation["critic_pad_mask"] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Add features for critic tokens
        if "critic_tokens" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["critic_tokens"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        if "critic_pad_mask" not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION]["critic_pad_mask"] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            advantage_scaling=getattr(config, "advantage_scaling", 1.0)
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        # Add Critic Tokenizer Step
        CriticTokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            task_key="critic_prompt"
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
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
