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

import gc
import os
import time
from collections import defaultdict, deque
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2better import MolmoAct2BetterConfig, MolmoAct2BetterPolicy
from lerobot.utils.constants import OBS_STATE, POLICY_PREPROCESSOR_DEFAULT_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = REPO_ROOT / "examples/experiments/configs/baseline_molmoact2_so101_002000.yaml"
RUN_INTEGRATION = os.environ.get("LEROBOT_RUN_MOLMOACT2BETTER_INTEGRATION") == "1"


class _SelectActionHarnessMixin:
    def __init__(self, *, batch_size: int = 4, chunk_size: int = 3, action_dim: int = 2):
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            n_action_steps=chunk_size,
            enable_inference_cuda_graph=False,
            train_action_expert_only=False,
        )
        self._action_queues = defaultdict(deque)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.predict_calls = 0

    def _rtc_enabled(self) -> bool:
        return False

    def _model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {"input_ids": batch["input_ids"]}

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:  # noqa: ARG002
        self.predict_calls += 1
        time.sleep(0.002)
        values = torch.arange(
            self.batch_size * self.chunk_size * self.action_dim,
            dtype=torch.float32,
        )
        return values.reshape(self.batch_size, self.chunk_size, self.action_dim)


class _OriginalSelectActionHarness(_SelectActionHarnessMixin, MolmoAct2Policy):
    pass


class _BetterSelectActionHarness(_SelectActionHarnessMixin, MolmoAct2BetterPolicy):
    pass


def test_select_action_refills_batched_queues_once_and_preserves_actions():
    batch_size = 4
    batch = {"input_ids": torch.zeros(batch_size, 1, dtype=torch.long)}
    original = _OriginalSelectActionHarness(batch_size=batch_size)
    better = _BetterSelectActionHarness(batch_size=batch_size)

    original.train()
    better.train()

    t0 = time.perf_counter()
    original_actions = original.select_action(batch)
    original_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    better_actions = better.select_action(batch)
    better_elapsed = time.perf_counter() - t0

    assert original.predict_calls == batch_size
    assert better.predict_calls == 1
    assert better_elapsed < original_elapsed
    assert torch.equal(better_actions, original_actions)
    assert original.training
    assert not better.training


def test_molmoact2better_factory_registration():
    from lerobot.policies.factory import get_policy_class, make_policy_config

    assert make_policy_config("molmoact2better").type == "molmoact2better"
    assert get_policy_class("molmoact2better") is MolmoAct2BetterPolicy


def _load_baseline_checkpoint_and_config() -> tuple[Path, dict[str, Any]]:
    yaml = pytest.importorskip("yaml")
    with BASELINE_CONFIG.open() as f:
        raw_config = yaml.safe_load(f)
    checkpoint = Path(raw_config["pretrained_name_or_path"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = REPO_ROOT / checkpoint
    return checkpoint, raw_config


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def _raw_molmoact2_observation(config: MolmoAct2BetterConfig) -> dict[str, Any]:
    state_dim = int(config.output_features["action"].shape[0])
    raw: dict[str, Any] = {
        OBS_STATE: torch.linspace(-0.25, 0.25, steps=state_dim),
        "task": "pick up the orange cube and place it in the bowl",
    }
    for key, feature in config.input_features.items():
        if not key.startswith("observation.images."):
            continue
        channels, height, width = (int(dim) for dim in feature.shape)
        raw[key] = torch.zeros(channels, height, width, dtype=torch.uint8)
    return raw


def _load_processed_baseline_batch(
    *,
    checkpoint: Path,
    config: MolmoAct2BetterConfig,
    device: str,
) -> dict[str, Any]:
    import lerobot.policies.molmoact2.processor_molmoact2  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        checkpoint,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        overrides={"device_processor": {"device": device}},
    )
    return preprocessor(_raw_molmoact2_observation(config))


def _seed_server_rng(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_like_policy_server_drtc(policy: MolmoAct2Policy, experiment_config: dict[str, Any]) -> None:
    policy.eval()
    with suppress(Exception):
        policy.config.use_amp = False

    if not bool(experiment_config.get("rtc_enabled", False)):
        return

    from lerobot.async_inference.rtc_guidance import AsyncRTCConfig, AsyncRTCProcessor

    max_gw_raw = experiment_config.get("rtc_max_guidance_weight")
    rtc_cfg = AsyncRTCConfig(
        enabled=True,
        prefix_attention_schedule=str(experiment_config.get("rtc_prefix_attention_schedule") or "linear"),
        max_guidance_weight=float(max_gw_raw) if max_gw_raw is not None else None,
        sigma_d=float(experiment_config.get("rtc_sigma_d") or 1.0),
        full_trajectory_alignment=bool(experiment_config.get("rtc_full_trajectory_alignment", False)),
    )
    rtc = AsyncRTCProcessor(rtc_cfg, postprocess=None)
    policy.rtc_processor = rtc
    model_value = getattr(policy, "model", None)
    if model_value is not None:
        model_value.rtc_processor = rtc
    with suppress(Exception):
        policy.config.rtc_config = type("RTCConfigShim", (), {"enabled": True})()


def _server_style_get_action_chunk(
    policy: MolmoAct2Policy,
    observation: dict[str, Any],
    *,
    actions_per_chunk: int,
    **kwargs: Any,
) -> Tensor:
    with torch.no_grad():
        chunk = policy.predict_action_chunk(observation, **kwargs)
    if chunk.ndim != 3:
        chunk = chunk.unsqueeze(0)
    return chunk[:, :actions_per_chunk, :]


def _timed_server_style_get_action_chunk(
    *,
    policy_cls: type[MolmoAct2Policy],
    config: MolmoAct2BetterConfig,
    checkpoint: Path,
    batch: dict[str, Any],
    device: str,
    experiment_config: dict[str, Any],
) -> tuple[Tensor, float]:
    policy = policy_cls.from_pretrained(checkpoint, config=config, strict=False).eval()
    _configure_like_policy_server_drtc(policy, experiment_config)
    actions_per_chunk = int(experiment_config.get("actions_per_chunk") or config.n_action_steps)

    for warmup_idx in range(2):
        _seed_server_rng(1000 + warmup_idx)
        warmup_actions = _server_style_get_action_chunk(
            policy,
            _clone_batch(batch),
            actions_per_chunk=actions_per_chunk,
        )
        del warmup_actions

    _seed_server_rng(1234)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    actions = _server_style_get_action_chunk(
        policy,
        _clone_batch(batch),
        actions_per_chunk=actions_per_chunk,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    actions = actions.detach().cpu()
    del policy
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return actions, elapsed


@pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="set LEROBOT_RUN_MOLMOACT2BETTER_INTEGRATION=1 to load the local MolmoAct2 checkpoint",
)
def test_checkpoint_predict_action_chunk_parity_and_timing(record_property):
    pytest.importorskip("safetensors")
    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("MolmoAct2 checkpoint timing test requires CUDA.")

    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    checkpoint, experiment_config = _load_baseline_checkpoint_and_config()
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip(f"MolmoAct2 checkpoint not found at {checkpoint}")

    device = "cuda"
    old_config = MolmoAct2Config.from_pretrained(checkpoint)
    better_config = MolmoAct2BetterConfig.from_pretrained(checkpoint)
    for config in (old_config, better_config):
        config.device = device
        config.pretrained_path = checkpoint
        config.inference_action_mode = "continuous"
        config.num_inference_steps = int(experiment_config.get("num_flow_matching_steps") or 5)
        config.enable_inference_cuda_graph = False

    batch = _load_processed_baseline_batch(
        checkpoint=checkpoint,
        config=better_config,
        device=device,
    )
    old_actions, old_elapsed = _timed_server_style_get_action_chunk(
        policy_cls=MolmoAct2Policy,
        config=old_config,
        checkpoint=checkpoint,
        batch=batch,
        device=device,
        experiment_config=experiment_config,
    )
    better_actions, better_elapsed = _timed_server_style_get_action_chunk(
        policy_cls=MolmoAct2BetterPolicy,
        config=better_config,
        checkpoint=checkpoint,
        batch=batch,
        device=device,
        experiment_config=experiment_config,
    )

    record_property("molmoact2_predict_action_chunk_seconds", old_elapsed)
    record_property("molmoact2better_predict_action_chunk_seconds", better_elapsed)
    torch.testing.assert_close(better_actions, old_actions, atol=1e-3, rtol=1e-3)
