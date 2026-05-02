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

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.policies.tinypi05.configuration_tinypi05 import TinyPI05Config
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
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="tinypi05_action_encoding_processor_step")
@dataclass
class TinyPi05ActionEncodingProcessorStep(ProcessorStep):
    """Convert an absolute action chunk into anchor- or delta-encoded targets.

    Runs **before** ``NormalizerProcessorStep`` in the preprocessor pipeline so the
    normalizer sees the encoded distribution (matched against
    ``compute_delta_stats.py --encoding {anchor,delta}``).

    Encoding (per element of an action chunk ``a_0, ..., a_{T-1}`` and current state ``s_0``):

    * ``"anchor"``: ``d_t = a_t - s_0`` for all t. Translation-invariant; recommended.
    * ``"delta"``:  ``d_0 = a_0 - s_0``, then ``d_t = a_t - a_{t-1}`` for t > 0.
      Drift-prone, but only ``d_0`` references the anchor (useful for RTC chunk splicing).
    * ``"absolute"``: no-op.

    Notes:

    * If ``ACTION`` is missing from the transition (e.g. inference time, where the
      preprocessor is run on observations only), this step is a no-op. Inference-time
      decoding back to absolute joint targets is **not** done here -- the postprocessor
      pipeline only sees the action tensor and has no observation context. Decode in
      the policy's ``predict_action_chunk`` instead, or in your inference loop, using
      the raw current state.
    """

    action_encoding: str = "absolute"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.action_encoding == "absolute":
            return transition

        actions = transition.get(TransitionKey.ACTION)
        if actions is None:
            return transition  # inference-time call: no action to encode.

        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError(
                f"action_encoding={self.action_encoding!r} requires {OBS_STATE!r} in the "
                "observation, but it was not found."
            )

        # actions: [B, T, A] absolute joint targets. state: [B, S] absolute joint positions.
        # Anchor on the first action_dim columns of state (assumes ACTION dims align with the
        # first dims of OBS_STATE, which holds for SO-100/SO-101 follower setups).
        action_dim = actions.shape[-1]
        anchor = state[..., :action_dim].to(dtype=actions.dtype)

        if self.action_encoding == "anchor":
            actions_enc = actions - anchor.unsqueeze(-2)
        elif self.action_encoding == "delta":
            d0 = (actions[:, 0, :] - anchor).unsqueeze(1)
            if actions.shape[1] > 1:
                drest = actions[:, 1:, :] - actions[:, :-1, :]
                actions_enc = torch.cat([d0, drest], dim=1)
            else:
                actions_enc = d0
        else:
            raise ValueError(f"Unknown action_encoding: {self.action_encoding!r}")

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = actions_enc
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def _load_encoded_action_stats(path: str) -> dict[str, torch.Tensor]:
    """Load `.pt` produced by `compute_delta_stats.py` and return a {stat_name: tensor} dict.

    The script writes ``{"min": ..., "max": ..., "mean": ..., "std": ..., "q01": ..., "q99": ...}``
    where each value has shape ``[chunk_size, action_dim]``. The ``NormalizerProcessorStep``
    QUANTILES path consumes ``q01`` and ``q99``; broadcasting against ``[B, T, A]`` actions works.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"action_encoding_stats_path not found: {path}")
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, dict):
        raise ValueError(
            f"Expected a dict from {path!r}; got {type(stats).__name__}. "
            "Regenerate via compute_delta_stats.py."
        )
    return stats


def make_tinypi05_pre_post_processors(
    config: TinyPI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create PI0.5-compatible processors for TinyPI05.

    When ``config.action_encoding`` is ``"anchor"`` or ``"delta"``:

    1. ``TinyPi05ActionEncodingProcessorStep`` is inserted before the normalizer to
       rewrite the action chunk in encoded coordinates using the current state.
    2. The action stats in ``dataset_stats[ACTION]`` are overridden with the encoded
       stats loaded from ``config.action_encoding_stats_path`` so the normalizer
       sees the correct (encoded) distribution.

    The postprocessor unnormalizer also picks up the overridden encoded stats; its
    output is therefore in **encoded** space, not absolute joint targets. Inference
    callers must decode back to absolute themselves (see the docstring on
    ``TinyPi05ActionEncodingProcessorStep``).
    """
    if config.action_encoding != "absolute":
        encoded_action_stats = _load_encoded_action_stats(config.action_encoding_stats_path)
        dataset_stats = dict(dataset_stats or {})
        dataset_stats[ACTION] = encoded_action_stats
        logging.info(
            "TinyPI05: using %s action encoding with stats from %s (per-timestep shape %s)",
            config.action_encoding,
            config.action_encoding_stats_path,
            tuple(encoded_action_stats["q01"].shape) if "q01" in encoded_action_stats else "?",
        )

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        # Action encoding (absolute -> anchor/delta) MUST run BEFORE the normalizer:
        # it consumes the raw absolute action chunk and the raw absolute OBS_STATE.
        TinyPi05ActionEncodingProcessorStep(action_encoding=config.action_encoding),
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
