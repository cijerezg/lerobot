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
"""Point-map encoder asserts (depth_pointmap_design.md) — CPU only.

What a real run can't catch cheaply: wrong back-projection math trains on garbage
geometry; deadzone/far pixels leaking in as fake surface; empty patches not routed
to the null bank; the recentering losing translation-invariance (the property that
makes the 2D conv metric-aware without voxelizing). Bit-identity at gate=0 with the
real read is verified in Phase 2 against the checkpoint, not here.
"""

import pytest
import torch
from torch import nn

from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig
from lerobot.policies.depth_pointmap.modeling_pointmap import (
    DepthPointmapEncoder,
    back_project,
    patchify,
)
from lerobot.rl.shared_config import MemoryConfig

# Camera looking along +z: pixel (u, v) sees ray ((u-cx)/fx, (v-cy)/fy, 1).
INTRINSICS = (100.0, 100.0, 32.0, 32.0)


def _bp(depth, **kw):
    return back_project(depth, intrinsics=INTRINSICS, depth_units_mm=1.0, z_min_mm=70.0, z_max_mm=800.0, **kw)


def test_back_project_metric_coords_and_mask():
    # Fronto-parallel plane at 200 mm. Center pixel (cx, cy) → X=Y=0, Z=200.
    depth = torch.full((1, 64, 64), 200.0)
    pm = _bp(depth)
    assert pm.shape == (1, 4, 64, 64)
    x, y, z, m = pm[0]
    assert torch.allclose(z, torch.full((64, 64), 200.0))
    assert m.sum() == 64 * 64
    assert torch.allclose(x[32, 32], torch.tensor(0.0))
    # pixel u=42 (10 px right of cx): X = (42-32)*200/100 = 20 mm.
    assert torch.allclose(x[32, 42], torch.tensor(20.0))
    assert torch.allclose(y[42, 32], torch.tensor(20.0))


def test_back_project_deadzone_far_and_holes_are_invalid():
    assert _bp(torch.full((1, 8, 8), 50.0))[0, 3].sum() == 0  # below z_min
    assert _bp(torch.full((1, 8, 8), 0.0))[0, 3].sum() == 0  # holes
    assert _bp(torch.full((1, 8, 8), 900.0))[0, 3].sum() == 0  # beyond z_max


def test_back_project_zeroes_invalid_coords():
    # An invalid pixel must carry zero coords, not garbage, so it cannot pollute a
    # masked centroid even if a downstream sum forgets to mask.
    depth = torch.full((1, 8, 8), 200.0)
    depth[0, 0, 0] = 0.0  # one hole
    pm = _bp(depth)
    assert torch.equal(pm[0, :, 0, 0], torch.zeros(4))


def test_uint16_units():
    raw = torch.full((1, 8, 8), 2000, dtype=torch.uint16)  # 0.1 mm/level → 200 mm
    pm = back_project(raw, intrinsics=INTRINSICS, depth_units_mm=0.1, z_min_mm=70.0, z_max_mm=800.0)
    assert torch.allclose(pm[0, 2], torch.full((8, 8), 200.0))


def test_patchify_order_and_shape():
    # Tag each patch with its row-major index in channel 0; patchify must preserve it.
    pm = torch.zeros(1, 1, 80, 120)  # 2 rows × 3 cols of 40×40 patches
    for r in range(2):
        for c in range(3):
            pm[0, 0, r * 40 : (r + 1) * 40, c * 40 : (c + 1) * 40] = r * 3 + c
    patches = patchify(pm, 40)
    assert patches.shape == (1, 6, 1, 40, 40)
    assert torch.equal(patches[0, :, 0, 0, 0], torch.arange(6.0))


def test_token_count_and_shape():
    cfg = DepthPointmapConfig()  # 480×640, P=40 → 192 tokens
    enc = DepthPointmapEncoder(cfg, d_mem=16)
    assert enc.num_tokens == 192
    depth = torch.full((2, 480, 640), 300.0)
    tokens = enc(_bp(depth))
    assert tokens.shape == (2, 192, 16)


def test_empty_patch_becomes_null_token():
    cfg = DepthPointmapConfig()
    enc = DepthPointmapEncoder(cfg, d_mem=16)
    depth = torch.full((1, 480, 640), 300.0)
    depth[:, :40, :40] = 0.0  # patch index 0 entirely holes → empty
    tokens = enc(_bp(depth))
    assert torch.allclose(tokens[0, 0], enc.null_tokens[0])
    assert not torch.allclose(tokens[0, -1], enc.null_tokens[-1])  # a full patch is not null


def test_shape_feature_is_translation_invariant():
    # Recentering removes position: a patch and the same shape shifted by a constant
    # 3D vector must yield the same shape feature f. (Disable the absolute-depth
    # channel, which is deliberately depth-dependent, and zero the position branch.)
    cfg = DepthPointmapConfig(image_size=(40, 40), patch_size=40, include_centroid_depth=False)
    enc = DepthPointmapEncoder(cfg, d_mem=16).eval()
    nn.init.zeros_(enc.pos_proj.weight)
    nn.init.zeros_(enc.pos_proj.bias)
    with torch.no_grad():
        enc.modality_embed.zero_()

    torch.manual_seed(0)
    coords = torch.randn(1, 3, 40, 40) * 10 + torch.tensor([0.0, 0.0, 300.0])[None, :, None, None]
    mask = torch.ones(1, 1, 40, 40)
    pm1 = torch.cat([coords, mask], dim=1)
    pm2 = torch.cat([coords + torch.tensor([15.0, -7.0, 22.0])[None, :, None, None], mask], dim=1)
    assert torch.allclose(enc(pm1), enc(pm2), atol=1e-5)


def test_encoder_is_gate_free():
    # The read gate lives on the DepthStream, not the encoder (the encoder just tokenizes).
    enc = DepthPointmapEncoder(DepthPointmapConfig(), d_mem=16)
    assert not hasattr(enc, "gate")
    assert not hasattr(enc, "abstain_bias")
    # memory_from_batch returns just the tokens (no gate tuple).
    mem = enc.memory_from_batch({}, batch_size=2, device=torch.device("cpu"))  # depth missing → null bank
    assert mem.shape == (2, enc.num_tokens, 16)


# --- Depth stream blocks -----------------------------------------------------------
# CRITIC-ONLY: the actor's depth now enters the VLM prefix as placeholder tokens, so the
# co-evolving DepthStream aggregate, its per-layer joint-softmax read (join_depth_columns,
# depth_attention_mass, depth_bias) and slice_wrist_cam_kv are gone. What survives is the
# block itself, which rl_molmoact2.py still stacks for the value function.

from lerobot.policies.depth_pointmap.modeling_stream import (  # noqa: E402
    DepthStreamBlock,
    mask_camera_patch_span,
)

D_VLM = 24
N_TOK = 6
B = 2


def _block(heads=4, d_d=16):
    return DepthStreamBlock(d_d=d_d, d_vlm=D_VLM, num_heads=heads, mlp_ratio=4.0)


def _wrist(t_w=5):
    return torch.randn(B, t_w, D_VLM), torch.randn(B, t_w, D_VLM)


def test_block_cross_on_kills_wrist_bridge():
    """RGB dropout masks the wrist span out of attention, but the block's K/V gather
    bypasses attention masks - cross_on=False must make the row's output independent
    of the wrist content."""
    torch.manual_seed(0)
    blk = _block().eval()
    init = torch.randn(B, N_TOK, 16)
    k_a, v_a = _wrist()
    k_b, v_b = k_a.clone(), v_a.clone()
    k_b[1] = torch.randn_like(k_b[1]) * 100  # garbage wrist content for row 1 only
    v_b[1] = torch.randn_like(v_b[1]) * 100
    cross_on = torch.tensor([True, False])
    out_a = blk(init, k_a, v_a, cross_on=cross_on)
    out_b = blk(init, k_b, v_b, cross_on=cross_on)
    assert torch.allclose(out_a[1], out_b[1])  # bridge-killed row ignores wrist content
    assert torch.allclose(out_a[0], out_b[0])  # row 0 (same content) unchanged
    out_on = blk(init, k_b, v_b, cross_on=torch.tensor([True, True]))
    assert not torch.allclose(out_on[1], out_a[1])  # bridge actually matters when on


def test_block_gradient_flows_to_all_params():
    blk = _block()
    init = torch.randn(B, N_TOK, 16, requires_grad=True)
    blk(init, *_wrist()).sum().backward()
    assert blk.self_attn.q_proj.weight.grad is not None
    assert blk.cross_attn.k_proj.weight.grad is not None


def test_rgb_dropout_masks_only_that_camera_span():
    """The processor's RGB-dropout edit: zero the dropped camera's <im_patch>
    columns of attention_mask for dropped rows only, leaving every other token
    (text, the other camera, padding state) untouched."""
    pid = 99
    # 2 cams x 3 patch tokens, text on both sides; row 1 is the dropped sample.
    input_ids = torch.tensor(
        [[5, 99, 99, 99, 7, 99, 99, 99, 8], [5, 99, 99, 99, 7, 99, 99, 99, 8]]
    )
    attention_mask = torch.ones(2, 9, dtype=torch.long)
    dropped = torch.tensor([False, True])
    rows = dropped.nonzero(as_tuple=True)[0]
    attention_mask = mask_camera_patch_span(
        attention_mask,
        input_ids,
        image_patch_id=pid,
        num_images=2,
        cam_index=1,
        rows=rows,
    )

    assert attention_mask[0].tolist() == [1] * 9  # undropped row untouched
    # row 1: cam1 span (positions 5,6,7) masked; text + cam0 span still visible
    assert attention_mask[1].tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 1]


def test_camera_token_meta_order_matches_the_processor():
    """cam_index is positional, so a mismatch silently ablates the wrong camera.
    Both branches of the resolution (dataset image_keys, or image_features when the
    policy is built from scratch) must yield the same bare names in the same order,
    and _pointmap_wrist_meta must agree with them."""
    import types

    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    class Stub:
        _camera_token_meta = MolmoAct2Policy._camera_token_meta
        _pointmap_wrist_meta = MolmoAct2Policy._pointmap_wrist_meta

        def __init__(self, image_keys):
            self.config = types.SimpleNamespace(
                image_keys=image_keys,
                # deliberately unsorted: the processor sorts, so we must too
                image_features={"observation.images.wrist": 1, "observation.images.top": 1},
                pointmap_config=types.SimpleNamespace(depth_key="wrist"),
            )
            self.model = types.SimpleNamespace(config=types.SimpleNamespace(image_patch_id=7))

    from_features = Stub([])
    from_keys = Stub(["observation.images.top", "observation.images.wrist"])

    assert from_features._camera_token_meta() == (7, ["top", "wrist"])
    assert from_keys._camera_token_meta() == (7, ["top", "wrist"])
    # (image_patch_id, num_images, cam_index) — wrist is camera 1 either way
    assert from_features._pointmap_wrist_meta() == (7, 2, 1)
    assert from_keys._pointmap_wrist_meta() == (7, 2, 1)


def _history_cfg():
    return DepthPointmapConfig(
        image_size=(80, 80), patch_size=40, depth_units_mm=1.0, history_num_samples=2
    )


def test_encoder_history_token_count_invariant():
    """History fuses inside the patch CNN (depth_history_design.md): past frames
    change the current tokens' content, never the token count."""
    torch.manual_seed(0)
    enc = DepthPointmapEncoder(_history_cfg(), d_mem=16).eval()
    n = enc.num_tokens  # 4
    cpu = torch.device("cpu")

    current = torch.full((1, 80, 80), 300.0)
    near = torch.full((1, 2, 80, 80), 200.0)
    far = torch.full((1, 2, 80, 80), 500.0)
    mem_near = enc.memory_from_batch(
        {"observation.depth.wrist": current, "history.depth.wrist.depth": near},
        batch_size=1,
        device=cpu,
    )
    assert mem_near.shape == (1, n, 16)  # NOT (T_h+1)·n — past rows are dropped in the CNN
    assert enc.null_memory(2).shape == (2, n, 16)

    mem_far = enc.memory_from_batch(
        {"observation.depth.wrist": current, "history.depth.wrist.depth": far},
        batch_size=1,
        device=cpu,
    )
    assert not torch.allclose(mem_near, mem_far)  # history content reaches the tokens


def test_encoder_history_mask_equals_missing_window():
    """The shared history-dropout draw (history_images_mask False) masks the past
    keys, which must compute exactly what a missing window (cold RTC deque, plain
    offline eval) computes — train/test parity for the no-history case."""
    torch.manual_seed(0)
    enc = DepthPointmapEncoder(_history_cfg(), d_mem=16).eval()
    cpu = torch.device("cpu")

    current = torch.full((1, 80, 80), 300.0)
    window = torch.full((1, 2, 80, 80), 200.0)
    dropped = enc.memory_from_batch(
        {
            "observation.depth.wrist": current,
            "history.depth.wrist.depth": window,
            "history_images_mask": torch.tensor([False]),
        },
        batch_size=1,
        device=cpu,
    )
    missing = enc.memory_from_batch({"observation.depth.wrist": current}, batch_size=1, device=cpu)
    kept = enc.memory_from_batch(
        {
            "observation.depth.wrist": current,
            "history.depth.wrist.depth": window,
            "history_images_mask": torch.tensor([True]),
        },
        batch_size=1,
        device=cpu,
    )
    assert torch.allclose(dropped, missing, atol=1e-5)
    assert not torch.allclose(kept, missing)  # unmasked history is actually used


def test_encoder_valid_history_recovers_fully_empty_current_patch():
    """A complete current-frame hole uses the newest valid history rather than
    discarding the temporal fusion at the final null-bank routing step."""
    torch.manual_seed(0)
    enc = DepthPointmapEncoder(_history_cfg(), d_mem=16).eval()
    cpu = torch.device("cpu")
    current = torch.zeros((1, 80, 80))
    near = torch.full((1, 2, 80, 80), 200.0)
    far = torch.full((1, 2, 80, 80), 500.0)

    mem_near = enc.memory_from_batch(
        {"observation.depth.wrist": current, "history.depth.wrist.depth": near},
        batch_size=1,
        device=cpu,
    )
    mem_far = enc.memory_from_batch(
        {"observation.depth.wrist": current, "history.depth.wrist.depth": far},
        batch_size=1,
        device=cpu,
    )
    null = enc.null_memory(1)
    assert not torch.allclose(mem_near, null)
    assert not torch.allclose(mem_near, mem_far)

    dropped = enc.memory_from_batch(
        {
            "observation.depth.wrist": current,
            "history.depth.wrist.depth": near,
            "history_images_mask": torch.tensor([False]),
        },
        batch_size=1,
        device=cpu,
    )
    assert torch.allclose(dropped, null)


@pytest.mark.parametrize("chunk_rows", [0, 4, 2])
@pytest.mark.parametrize("history", [2, 0])
def test_encoder_checkpointing_changes_nothing_but_memory(history, chunk_rows):
    """Checkpointing the patch CNN, and chunking it over patch rows, must be exact.

    Chunking is only sound because every op in the trunk (convs, GroupNorm, the
    same-pixel temporal attention, the final spatial mean) is independent across
    patch rows; a future op that mixes rows would silently break this.
    """
    cfg = DepthPointmapConfig(
        image_size=(80, 80), patch_size=40, depth_units_mm=1.0,
        history_num_samples=history, dropout_prob=0.0, encoder_chunk_rows=chunk_rows,
    )
    batch = {"observation.depth.wrist": torch.rand(3, 80, 80) * 400 + 100}
    if history:
        batch["history.depth.wrist.depth"] = torch.rand(3, history, 80, 80) * 400 + 100
        batch["history_images_mask"] = torch.tensor([True, False, True])

    grads = []
    outs = []
    for gradient_checkpointing in (False, True):
        torch.manual_seed(0)
        enc = DepthPointmapEncoder(
            cfg, d_mem=16, gradient_checkpointing=gradient_checkpointing
        ).train()
        out = enc.memory_from_batch(batch, batch_size=3, device=torch.device("cpu"))
        out.pow(2).mean().backward()
        outs.append(out.detach())
        grads.append({n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None})

    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-6)
    assert grads[0].keys() == grads[1].keys() and grads[0]
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-4, atol=1e-7, msg=name)


def test_config_rejects_negative_history_samples():
    with pytest.raises(ValueError, match="history_num_samples"):
        DepthPointmapConfig(history_num_samples=-1)
    with pytest.raises(ValueError, match="history_num_samples"):
        MemoryConfig(history_num_samples=-1)
    with pytest.raises(ValueError, match="history_num_samples"):
        MemoryConfig(history_keys=["depth.wrist.depth"], history_num_samples=0)
