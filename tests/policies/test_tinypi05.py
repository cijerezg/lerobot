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

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.tinypi05 import TinyPI05Config, TinyPI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

pytest.importorskip("transformers")


def _debug_config() -> TinyPI05Config:
    config = TinyPI05Config(
        architecture_preset="debug",
        image_resolution=(32, 32),
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=8,
        max_action_dim=8,
        tokenizer_max_length=16,
        num_inference_steps=1,
        dtype="float32",
        gradient_checkpointing=False,
        push_to_hub=False,
    )
    config.input_features = {
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return config


def _batch(config: TinyPI05Config, device: torch.device | str | None = None) -> dict[str, torch.Tensor]:
    batch_size = 1
    return {
        "observation.images.top": torch.rand(batch_size, 3, 32, 32, device=device),
        OBS_STATE: torch.zeros(batch_size, 6, device=device),
        ACTION: torch.zeros(batch_size, config.chunk_size, 6, device=device),
        OBS_LANGUAGE_TOKENS: torch.ones(
            batch_size, config.tokenizer_max_length, dtype=torch.long, device=device
        ),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(
            batch_size, config.tokenizer_max_length, dtype=torch.bool, device=device
        ),
    }


def test_tinypi05_factory_registration():
    assert get_policy_class("tinypi05") is TinyPI05Policy
    assert isinstance(make_policy_config("tinypi05", architecture_preset="debug"), TinyPI05Config)


def test_tinypi05_forward_and_predict_shapes():
    config = _debug_config()
    policy = TinyPI05Policy(config)
    batch = _batch(config, device=config.device)

    loss, loss_dict = policy.forward(batch)
    assert loss.ndim == 0
    assert "loss" in loss_dict

    actions = policy.predict_action_chunk(batch)
    assert actions.shape == (1, config.chunk_size, 6)


def test_tinypi05_processors(monkeypatch):
    class DummyTokenizer:
        def __call__(self, text, max_length, truncation, padding, padding_side, return_tensors):
            batch_size = len(text)
            return {
                "input_ids": torch.ones(batch_size, max_length, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
            }

    import lerobot.processor.tokenizer_processor as tokenizer_processor

    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: DummyTokenizer(),
    )

    config = _debug_config()
    stats = {
        OBS_STATE: {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
            "min": torch.zeros(6),
            "max": torch.ones(6),
            "q01": -torch.ones(6),
            "q99": torch.ones(6),
        },
        ACTION: {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
            "min": torch.zeros(6),
            "max": torch.ones(6),
            "q01": -torch.ones(6),
            "q99": torch.ones(6),
        },
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)

    processed = preprocessor(
        {
            "observation.images.top": torch.rand(3, 32, 32),
            OBS_STATE: torch.zeros(6),
            ACTION: torch.zeros(config.chunk_size, 6),
            "task": "Pick up the orange cube",
        }
    )
    assert OBS_LANGUAGE_TOKENS in processed
    assert OBS_LANGUAGE_ATTENTION_MASK in processed

    action = postprocessor(torch.zeros(1, config.chunk_size, 6))
    assert action.shape == (1, config.chunk_size, 6)
