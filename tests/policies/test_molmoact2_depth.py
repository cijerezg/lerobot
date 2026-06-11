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
"""CPU-only coverage for the external RealSense depth integration (depth_integration_plan.md).

These exercise the pure pieces of all three tracks without the HF checkpoint or a GPU:
delivery adapters (Track A), the trainer's complementary_info lift (Track B), and the
config / normalizer / encoder (Track C). The end-to-end forward + train/infer smoke on the
real checkpoint still needs a GPU (see the plan's "Open items").
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2DepthEncoder
from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2DepthNormalizerProcessorStep
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2Critic
from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
from lerobot.types import TransitionKey
from lerobot.utils.feature_utils import build_dataset_frame, resolve_depth_keys

DS_FEATURES = {
    "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["h", "w", "c"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
}


def _raw_obs():
    return {
        "top": np.zeros((480, 640, 3), dtype=np.uint8),
        "top.depth": np.full((480, 640), 1234, dtype=np.uint16),
        "a": 0.1,
        "b": 0.2,
    }


# --------------------------------------------------------------------------- Track C: config
def test_config_no_op_defaults():
    cfg = MolmoAct2Config()
    assert cfg.enable_depth is False
    assert cfg.depth_keys == []


def test_config_validation_requires_depth_keys():
    with pytest.raises(ValueError, match="non-empty depth_keys"):
        MolmoAct2Config(enable_depth=True, depth_keys=[])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"depth_clip_mm": 0.0}, "depth_clip_mm"),
        ({"depth_units_mm": -1.0}, "depth_units_mm"),
        ({"depth_num_tokens": 0}, "depth_num_tokens"),
    ],
)
def test_config_validation_positive_fields(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MolmoAct2Config(enable_depth=True, depth_keys=["top"], **kwargs)


# --------------------------------------------------------------------------- Track C: normalizer
def _run_norm(step, depth):
    transition = {TransitionKey.OBSERVATION: {"observation.depth.top": depth}}
    out = step(transition)
    return out[TransitionKey.OBSERVATION]["observation.depth.top"]


def test_normalizer_disabled_is_passthrough():
    step = MolmoAct2DepthNormalizerProcessorStep(enable_depth=False, depth_keys=["top"])
    depth = torch.full((1, 1, 4, 4), 1234.0)
    transition = {TransitionKey.OBSERVATION: {"observation.depth.top": depth}}
    assert step(transition)[TransitionKey.OBSERVATION]["observation.depth.top"] is depth


def test_normalizer_range_shape_and_holes():
    step = MolmoAct2DepthNormalizerProcessorStep(
        enable_depth=True, depth_keys=["top"], depth_units_mm=0.1, depth_clip_mm=1000.0
    )
    # raw 5000 -> 500 mm -> 0.5; raw 0 (hole) -> 0; raw 50000 -> 5000 mm -> clamp 1000 -> 1.0
    raw = torch.tensor([[0.0, 5000.0], [50000.0, 5000.0]])  # (H, W), exercises the (2)->(B,1,H,W) path
    out = _run_norm(step, raw)
    assert out.shape == (1, 1, 2, 2)
    assert out.dtype == torch.float32
    assert out[0, 0, 0, 0].item() == 0.0  # hole stays 0
    assert abs(out[0, 0, 0, 1].item() - 0.5) < 1e-6
    assert out[0, 0, 1, 0].item() == 1.0  # clamped
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalizer_batched_channel_dim():
    step = MolmoAct2DepthNormalizerProcessorStep(
        enable_depth=True, depth_keys=["top"], depth_units_mm=0.1, depth_clip_mm=1000.0
    )
    out = _run_norm(step, torch.full((3, 8, 8), 5000.0))  # (B, H, W) -> (B, 1, H, W)
    assert out.shape == (3, 1, 8, 8)


# --------------------------------------------------------------------------- Track C: encoder
def test_encoder_zero_init_is_exact_zero():
    enc = MolmoAct2DepthEncoder(hidden_size=16, num_tokens=2, gate_init_bias=-4.0)
    tokens = enc(torch.rand(3, 1, 48, 64))
    assert tokens.shape == (3, 2, 16)
    assert float(tokens.detach().abs().max()) == 0.0  # zero-init proj => exact-zero tokens at init


def test_encoder_grads_reach_all_params_including_gate():
    enc = MolmoAct2DepthEncoder(hidden_size=16, num_tokens=1, gate_init_bias=-4.0)
    # Push a non-zero target through so the zero-init proj still receives gradient.
    loss = (enc(torch.rand(2, 1, 32, 32)) - 1.0).pow(2).mean()
    loss.backward()
    missing = [name for name, p in enc.named_parameters() if p.grad is None]
    assert not missing, f"params with no grad: {missing}"


# --------------------------------------------------------------------------- Track A: delivery
def test_build_frame_drop_when_disabled():
    frame = build_dataset_frame(DS_FEATURES, _raw_obs(), prefix="observation", depth_keys=None)
    assert set(frame) == {"observation.images.top", "observation.state"}


def test_build_frame_carry_when_enabled():
    frame = build_dataset_frame(DS_FEATURES, _raw_obs(), prefix="observation", depth_keys=["top"])
    d = frame["observation.depth.top"]
    assert d.dtype == np.uint16 and d.shape == (480, 640)


def test_prepare_observation_depth_branch():
    frame = build_dataset_frame(DS_FEATURES, _raw_obs(), prefix="observation", depth_keys=["top"])
    out = prepare_observation_for_inference(frame, torch.device("cpu"), task="t", robot_type="r")
    d = out["observation.depth.top"]
    assert d.dtype == torch.float32 and d.shape == (1, 1, 480, 640)
    assert float(d.max()) == 1234.0  # raw-mm preserved (no /255 image branch, no normalize)


def test_resolve_depth_keys_gate():
    assert resolve_depth_keys(SimpleNamespace(enable_depth=True, depth_keys=["top"])) == ["top"]
    assert resolve_depth_keys(SimpleNamespace(enable_depth=False, depth_keys=["top"])) is None
    assert resolve_depth_keys(SimpleNamespace()) is None  # non-depth policy: getattr guard


# --------------------------------------------------------------------------- Track B: trainer lift
def _cfg(enable_depth, depth_keys):
    return SimpleNamespace(policy=SimpleNamespace(enable_depth=enable_depth, depth_keys=depth_keys))


def test_inject_depth_disabled_is_no_op():
    obs = {"observation.state": torch.zeros(2, 2)}
    comp = {"depth.top.depth": torch.zeros(2, 8, 8, dtype=torch.uint16)}
    out = MolmoAct2Trainer._inject_depth_observations(obs, comp, _cfg(False, ["top"]))
    assert "observation.depth.top" not in out


def test_inject_depth_lifts_and_shapes():
    obs = {"observation.state": torch.zeros(2, 2)}
    comp = {"depth.top.depth": torch.full((2, 8, 8), 1234, dtype=torch.uint16)}
    out = MolmoAct2Trainer._inject_depth_observations(obs, comp, _cfg(True, ["top"]))
    d = out["observation.depth.top"]
    assert d.shape == (2, 1, 8, 8) and d.dtype == torch.float32
    assert float(d.max()) == 1234.0  # raw; normalize happens later in the preprocessor


def test_inject_depth_skips_cam_not_in_depth_keys():
    obs = {}
    comp = {"depth.wrist.depth": torch.zeros(2, 8, 8, dtype=torch.uint16)}
    out = MolmoAct2Trainer._inject_depth_observations(obs, comp, _cfg(True, ["top"]))
    assert out == {}


def test_inject_depth_prefixes_select_current_vs_next():
    comp = {
        "depth.top.depth": torch.full((2, 8, 8), 100, dtype=torch.uint16),
        "next_depth.top.depth": torch.full((2, 8, 8), 200, dtype=torch.uint16),
    }
    curr = MolmoAct2Trainer._inject_depth_observations({}, comp, _cfg(True, ["top"]))
    assert float(curr["observation.depth.top"].max()) == 100.0
    nxt = MolmoAct2Trainer._inject_depth_observations({}, comp, _cfg(True, ["top"]), key_prefix="next_depth.")
    assert float(nxt["observation.depth.top"].max()) == 200.0


def test_rtc_obs_with_depth_gating_and_copy():
    from lerobot.rl.rtc_actor_runtime import _obs_with_depth

    env_obs = {"top.depth": torch.full((8, 8), 5, dtype=torch.uint16), "agent_pos": np.zeros(6)}
    policy_obs = {"observation.state": torch.zeros(6)}
    off = _obs_with_depth(policy_obs, env_obs, _cfg(False, ["top"]))
    assert off is policy_obs  # disabled: same object, untouched
    on = _obs_with_depth(policy_obs, env_obs, _cfg(True, ["top"]))
    assert torch.equal(on["observation.depth.top"], env_obs["top.depth"])
    assert "observation.depth.top" not in policy_obs  # shallow copy: caller's dict stays depth-free


def test_buffer_sample_emits_next_depth_aligned_with_next_state():
    from lerobot.rl.buffer import ReplayBuffer

    n, chunk = 12, 3
    buf = ReplayBuffer(
        capacity=n, device="cpu", state_keys=["observation.state"], storage_device="cpu",
        optimize_memory=True,
    )
    for i in range(n):
        buf.add(
            state={"observation.state": torch.full((1, 2), float(i))},
            action=torch.zeros(1, 2),
            reward=0.0,
            next_state={"observation.state": torch.full((1, 2), float(i + 1))},
            done=False,
            truncated=False,
            complementary_info={"depth.top.depth": torch.full((1, 8, 8), i, dtype=torch.uint16)},
        )
    batch = buf.sample(4, action_chunk_size=chunk)
    depth = batch["complementary_info"]["depth.top.depth"]
    next_depth = batch["complementary_info"]["next_depth.top.depth"]
    assert next_depth.shape == depth.shape and next_depth.dtype == torch.uint16
    # Depth values encode the frame index: next_depth must be the depth of the frame
    # next_state is derived from (idx + chunk, modulo capacity), per sample row.
    sampled_idx = depth[:, 0, 0].long()
    assert torch.equal(next_depth[:, 0, 0].long(), (sampled_idx + chunk) % n)
    # And that frame is exactly the one next_state reports.
    assert torch.equal(
        batch["next_state"]["observation.state"][:, 0].long(), (sampled_idx + chunk) % n
    )


# --------------------------------------------------------------------------- Critic depth encoder
def _critic_config(enable_depth, depth_keys):
    # MolmoAct2Critic.__init__ only needs these scalar fields (no backbone).
    return SimpleNamespace(
        num_value_bins=8,
        critic_llm_depth=2,
        dtype="float32",
        value_support_min=-2.0,
        value_support_max=0.0,
        hl_gauss_sigma_ratio=8.0,
        enable_depth=enable_depth,
        depth_keys=depth_keys,
        depth_num_tokens=1,
        depth_gate_init_bias=-4.0,
    )


def test_critic_depth_encoder_built_only_when_enabled():
    assert MolmoAct2Critic(_critic_config(False, [])).depth_encoder is None
    assert MolmoAct2Critic(_critic_config(True, ["top"])).depth_encoder is not None


def test_critic_depth_tokens_shape_and_zero_init():
    critic = MolmoAct2Critic(_critic_config(True, ["top"]))
    batch = {"observation.depth.top": torch.rand(3, 1, 48, 64)}
    tokens = critic.depth_tokens(batch, dtype=torch.float32, device=torch.device("cpu"))
    assert tokens.shape == (3, 1, MolmoAct2Critic.TEXT_HIDDEN_SIZE)
    assert float(tokens.detach().abs().max()) == 0.0  # zero-init proj + gate => exact-zero at init


def test_critic_depth_tokens_none_when_disabled_or_absent():
    assert MolmoAct2Critic(_critic_config(False, [])).depth_tokens(
        {"observation.depth.top": torch.rand(1, 1, 8, 8)}, dtype=torch.float32, device=torch.device("cpu")
    ) is None
    # Enabled but the configured cam is not in the batch -> None (no tokens appended).
    assert MolmoAct2Critic(_critic_config(True, ["top"])).depth_tokens(
        {"observation.depth.wrist": torch.rand(1, 1, 8, 8)}, dtype=torch.float32, device=torch.device("cpu")
    ) is None
