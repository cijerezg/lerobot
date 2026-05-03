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

"""Parity tests between `tinypi05` and `tinypi05v2` on a real checkpoint.

These tests assert that the self-contained `tinypi05v2` modeling layer produces
outputs that match the original `tinypi05` (which inherits from `pistar06`) when
both load the same 018000-step checkpoint. The tests are skipped gracefully
when the checkpoint directory is not present so unrelated CI jobs do not break.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("safetensors")
pytest.importorskip("transformers")  # tinypi05 (v1) requires transformers

from lerobot.policies.tinypi05 import TinyPI05Config, TinyPI05Policy  # noqa: E402
from lerobot.policies.tinypi05v2 import TinyPI05V2Config, TinyPI05V2Policy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

CKPT = Path(
    "outputs/train/2026-05-02/18-34-57_tinypi05_so101_pickplace_160_bs64_anchor/"
    "checkpoints/018000/pretrained_model"
)

# Loose tolerance: the checkpoint runs transformer layers in bfloat16, which
# accumulates ~1e-3-ish drift between different-but-equivalent attention
# kernels. Tight enough to catch algorithmic mismatches.
_PREFIX_ATOL = 5e-3
_DENOISE_ATOL = 5e-3
_ACTION_ATOL = 2e-3
_LOSS_ATOL = 5e-3
_EMBED_ATOL = 1e-4


@pytest.fixture(scope="module")
def policies():
    if not (CKPT / "model.safetensors").is_file():
        pytest.skip(f"tinypi05 checkpoint not found at {CKPT}")

    cfg_v1 = TinyPI05Config.from_pretrained(CKPT)
    cfg_v2 = TinyPI05V2Config.from_pretrained(CKPT)

    # Force CPU + float32 to keep the test deterministic and CI-friendly.
    cfg_v1.device = "cpu"
    cfg_v1.dtype = "bfloat16"  # keep checkpoint dtype so strict loading works
    cfg_v1.gradient_checkpointing = False

    cfg_v2.device = "cpu"
    cfg_v2.dtype = "bfloat16"
    cfg_v2.gradient_checkpointing = False

    v1 = TinyPI05Policy.from_pretrained(CKPT, config=cfg_v1).eval()
    v2 = TinyPI05V2Policy.from_pretrained(CKPT, config=cfg_v2).eval()
    return v1, v2


def _fake_batch(config_v1: TinyPI05Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    bsize = 1
    image_keys = list(config_v1.image_features)
    batch: dict[str, torch.Tensor] = {}
    for key in image_keys:
        batch[key] = torch.rand(bsize, 3, *config_v1.image_resolution)
    batch[OBS_STATE] = torch.zeros(bsize, config_v1.max_state_dim)
    # Random but deterministic language tokens (small vocab range).
    seq = config_v1.tokenizer_max_length
    tokens = torch.randint(0, 100, (bsize, seq), dtype=torch.long)
    masks = torch.ones(bsize, seq, dtype=torch.bool)
    batch[OBS_LANGUAGE_TOKENS] = tokens
    batch[OBS_LANGUAGE_ATTENTION_MASK] = masks
    return batch


def _preprocess_images(policy, batch):
    return policy._preprocess_images(batch)  # noqa: SLF001


def test_embed_image_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)
    images_v1, _ = _preprocess_images(v1, batch)
    images_v2, _ = _preprocess_images(v2, batch)

    for img_v1, img_v2 in zip(images_v1, images_v2, strict=True):
        # Image preprocessing is identical; the two models should see the same
        # pixel tensors.
        assert torch.allclose(img_v1, img_v2, atol=1e-6), "preprocessed images differ"

    for img in images_v1:
        out_v1 = v1.model.paligemma_with_expert.embed_image(img)
        out_v2 = v2.model.paligemma_with_expert.embed_image(img)
        assert out_v1.shape == out_v2.shape
        diff = (out_v1.float() - out_v2.float()).abs().max().item()
        assert diff < _EMBED_ATOL, f"embed_image max diff {diff} exceeds {_EMBED_ATOL}"


def test_embed_language_tokens_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    out_v1 = v1.model.paligemma_with_expert.embed_language_tokens(tokens).float()
    out_v2 = v2.model.paligemma_with_expert.embed_language_tokens(tokens).float()
    assert out_v1.shape == out_v2.shape
    diff = (out_v1 - out_v2).abs().max().item()
    assert diff < _EMBED_ATOL, f"embed_language_tokens max diff {diff} exceeds {_EMBED_ATOL}"


def test_prefix_forward_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)
    images_v1, img_masks_v1 = _preprocess_images(v1, batch)
    images_v2, img_masks_v2 = _preprocess_images(v2, batch)

    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Build identical prefix inputs via each policy's embed_prefix.
    prefix_embs_v1, prefix_pad_masks_v1, prefix_att_masks_v1 = v1.model.embed_prefix(
        images_v1, img_masks_v1, tokens, masks
    )
    prefix_embs_v2, prefix_pad_masks_v2, prefix_att_masks_v2 = v2.model.embed_prefix(
        images_v2, img_masks_v2, tokens, masks
    )

    assert torch.equal(prefix_pad_masks_v1, prefix_pad_masks_v2)
    assert torch.equal(prefix_att_masks_v1, prefix_att_masks_v2)
    diff_embs = (prefix_embs_v1.float() - prefix_embs_v2.float()).abs().max().item()
    assert diff_embs < _EMBED_ATOL, f"prefix embeddings diverge by {diff_embs}"

    # Run prefix-only forward on both, ensure last_hidden_state is close.
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as v1_make_att
    from lerobot.policies.tinypi05v2.modeling_tinypi05v2 import (
        _make_att_2d_masks,
        _prepare_attention_masks_4d,
    )

    att_2d_v1 = v1_make_att(prefix_pad_masks_v1, prefix_att_masks_v1)
    att_2d_v2 = _make_att_2d_masks(prefix_pad_masks_v2, prefix_att_masks_v2)
    assert torch.equal(att_2d_v1, att_2d_v2)
    position_ids = torch.cumsum(prefix_pad_masks_v1, dim=1) - 1
    att_2d_4d = _prepare_attention_masks_4d(att_2d_v2)

    with torch.no_grad():
        (pref_v1, _), _ = v1.model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_v1.clone(), None],
            use_cache=False,
        )
        (pref_v2, _), _ = v2.model.paligemma_with_expert.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs_v2.clone(), None],
            use_cache=False,
        )

    diff = (pref_v1.float() - pref_v2.float()).abs().max().item()
    assert diff < _PREFIX_ATOL, f"prefix hidden states diverge by {diff}"


def test_denoise_step_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)
    images_v1, img_masks_v1 = _preprocess_images(v1, batch)
    images_v2, img_masks_v2 = _preprocess_images(v2, batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Build prefix cache on v1 (uses transformers DynamicCache).
    prefix_embs_v1, prefix_pad_masks_v1, prefix_att_masks_v1 = v1.model.embed_prefix(
        images_v1, img_masks_v1, tokens, masks
    )
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks as v1_make_att

    att_2d_v1 = v1_make_att(prefix_pad_masks_v1, prefix_att_masks_v1)
    position_ids_v1 = torch.cumsum(prefix_pad_masks_v1, dim=1) - 1
    from lerobot.policies.tinypi05v2.modeling_tinypi05v2 import _prepare_attention_masks_4d

    with torch.no_grad():
        _, past_v1 = v1.model.paligemma_with_expert.forward(
            attention_mask=_prepare_attention_masks_4d(att_2d_v1),
            position_ids=position_ids_v1,
            past_key_values=None,
            inputs_embeds=[prefix_embs_v1, None],
            use_cache=True,
        )

    # Build prefix cache on v2 (uses list[(k,v)]).
    prefix_embs_v2, prefix_pad_masks_v2, prefix_att_masks_v2 = v2.model.embed_prefix(
        images_v2, img_masks_v2, tokens, masks
    )
    att_2d_v2 = v1_make_att(prefix_pad_masks_v2, prefix_att_masks_v2)
    position_ids_v2 = torch.cumsum(prefix_pad_masks_v2, dim=1) - 1
    with torch.no_grad():
        _, past_v2 = v2.model.paligemma_with_expert.forward(
            attention_mask=_prepare_attention_masks_4d(att_2d_v2),
            position_ids=position_ids_v2,
            past_key_values=None,
            inputs_embeds=[prefix_embs_v2, None],
            use_cache=True,
        )

    # Run a single denoise step with identical x_t and timestep.
    torch.manual_seed(0)
    x_t = torch.randn(1, v1.config.chunk_size, v1.config.max_action_dim)
    timestep = torch.tensor([0.5])

    with torch.no_grad():
        v_t_v1 = v1.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks_v1,
            past_key_values=past_v1,
            x_t=x_t,
            timestep=timestep,
        )
        v_t_v2 = v2.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks_v2,
            past_key_values=past_v2,
            x_t=x_t,
            timestep=timestep,
        )

    diff = (v_t_v1.float() - v_t_v2.float()).abs().max().item()
    assert diff < _DENOISE_ATOL, f"denoise_step outputs diverge by {diff}"


def test_predict_action_chunk_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)

    # Fix sampled noise to the same tensor for both models.
    fixed_noise_shape = (1, v1.config.chunk_size, v1.config.max_action_dim)
    torch.manual_seed(0)
    fixed_noise = torch.randn(*fixed_noise_shape, dtype=torch.float32)

    def patched_sample_noise(shape, device):  # noqa: ARG001
        return fixed_noise.to(device)

    orig_v1 = v1.model.sample_noise
    orig_v2 = v2.model.sample_noise
    v1.model.sample_noise = patched_sample_noise
    v2.model.sample_noise = patched_sample_noise
    try:
        with torch.no_grad():
            actions_v1 = v1.predict_action_chunk(batch)
            actions_v2 = v2.predict_action_chunk(batch)
    finally:
        v1.model.sample_noise = orig_v1
        v2.model.sample_noise = orig_v2

    assert actions_v1.shape == actions_v2.shape
    diff = (actions_v1.float() - actions_v2.float()).abs().max().item()
    assert diff < _ACTION_ATOL, f"predict_action_chunk diverges by {diff}"


def test_training_forward_parity(policies):
    v1, v2 = policies
    batch = _fake_batch(v1.config)

    # Inject a padded action target so training forward has something to train on.
    bsize = 1
    action_dim = v1.config.output_features["action"].shape[0]
    batch["action"] = torch.zeros(bsize, v1.config.chunk_size, action_dim)

    torch.manual_seed(0)
    noise = torch.randn(bsize, v1.config.chunk_size, v1.config.max_action_dim)
    time = torch.full((bsize,), 0.3)

    images_v1, img_masks_v1 = _preprocess_images(v1, batch)
    images_v2, img_masks_v2 = _preprocess_images(v2, batch)
    tokens = batch[OBS_LANGUAGE_TOKENS]
    masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    # Pad the action to max_action_dim to match what the underlying model expects.
    from lerobot.policies.pi05.modeling_pi05 import pad_vector as pad_v1

    actions_padded = pad_v1(batch["action"], v1.config.max_action_dim)

    with torch.no_grad():
        losses_v1 = v1.model.forward(
            images_v1, img_masks_v1, tokens, masks, actions_padded, noise=noise, time=time
        )
        losses_v2 = v2.model.forward(
            images_v2, img_masks_v2, tokens, masks, actions_padded, noise=noise, time=time
        )

    diff = (losses_v1.float() - losses_v2.float()).abs().max().item()
    assert diff < _LOSS_ATOL, f"training loss diverges by {diff}"
    # Also sanity-check overall loss magnitudes agree.
    assert math.isclose(
        losses_v1.mean().item(),
        losses_v2.mean().item(),
        abs_tol=_LOSS_ATOL,
    ), "mean training losses disagree beyond tolerance"
