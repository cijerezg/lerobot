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

import torch
from torch import nn

from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig
from lerobot.policies.depth_pointmap.modeling_pointmap import (
    DepthPointmapEncoder,
    back_project,
    patchify,
)

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


# --- MoT co-evolving depth stream (depth_pointmap_design.md Part B) ----------------

from lerobot.policies.depth_pointmap.modeling_stream import (  # noqa: E402
    DepthStream,
    gated_depth_read,
    slice_wrist_cam_kv,
)

D_VLM = 24
N_TOK = 6
B = 2


def _stream(num_layers=4, heads=2, head_dim=8, **cfg_kw):
    cfg = DepthPointmapConfig(stream_width=16, stream_num_heads=4, **cfg_kw)
    return DepthStream(
        cfg, d_vlm=D_VLM, num_action_heads=heads, action_head_dim=head_dim, num_layers=num_layers
    )


def _wrist_kv(num_layers, t_w=5):
    keys = [torch.randn(B, t_w, D_VLM) for _ in range(num_layers)]
    values = [torch.randn(B, t_w, D_VLM) for _ in range(num_layers)]
    return keys, values


def test_stream_emits_one_state_per_layer():
    s = _stream(num_layers=4)
    init = torch.randn(B, N_TOK, 16)
    states = s(init, *_wrist_kv(4))
    assert len(states) == 4
    assert all(state.shape == (B, N_TOK, 16) for state in states)


def test_stream_co_evolves():
    # Successive layer states must differ — that is the whole point of co-evolution.
    s = _stream(num_layers=3)
    init = torch.randn(B, N_TOK, 16)
    states = s(init, *_wrist_kv(3))
    assert not torch.allclose(states[0], states[1])
    assert not torch.allclose(states[1], states[2])


def test_stream_gate_and_sink_zero_init():
    s = _stream(num_layers=4, heads=2)
    assert s.gate.shape == (4,)
    assert s.sink_logit.shape == (4, 2)
    assert torch.equal(s.gate_value(), torch.zeros(4))  # tanh(0)=0 ⇒ depth read is bitwise zero at init
    assert torch.equal(s.sink_logit, torch.zeros(4, 2))


def test_stream_read_kv_into_action_head_space():
    s = _stream(num_layers=2, heads=3, head_dim=8)
    init = torch.randn(B, N_TOK, 16)
    state = s(init, *_wrist_kv(2))[0]
    k, v = s.read_kv(state)
    assert k.shape == (B, N_TOK, 3, 8)
    assert v.shape == (B, N_TOK, 3, 8)


def test_stream_rejects_wrong_layer_count():
    s = _stream(num_layers=4)
    init = torch.randn(B, N_TOK, 16)
    try:
        s(init, *_wrist_kv(3))  # 3 wrist layers for a 4-layer stream
    except ValueError as e:
        assert "wrist-cam KV layers" in str(e)
    else:
        raise AssertionError("expected a ValueError on layer-count mismatch")


def test_slice_wrist_cam_kv_picks_right_camera_span():
    # Two cameras × 3 patch tokens each (id=99), wrapped in text; cam 1 (the second
    # run) is the depth camera. Layout differs per row (variable left text) to prove
    # the per-row gather. d_vlm encodes the token's sequence position so we can check.
    pid = 99
    row0 = torch.tensor([5, 99, 99, 99, 7, 99, 99, 99, 8])  # cam0=pos1-3, cam1=pos5-7
    row1 = torch.tensor([5, 6, 99, 99, 99, 99, 99, 99, 8])  # cam0=pos2-4, cam1=pos5-7
    input_ids = torch.stack([row0, row1])
    t = input_ids.shape[1]
    # one layer; K = V = position index broadcast over d_vlm=2
    pos = torch.arange(t).float()[None, :, None].expand(2, t, 2).clone()
    keys, values = slice_wrist_cam_kv(
        [(pos, pos * 10)], input_ids=input_ids, image_patch_id=pid, num_images=2, cam_index=1
    )
    assert keys[0].shape == (2, 3, 2)
    # cam_index=1 → both rows' second run is positions 5,6,7
    assert torch.equal(keys[0][:, :, 0], torch.tensor([[5.0, 6, 7], [5, 6, 7]]))
    assert torch.equal(values[0][:, :, 0], torch.tensor([[50.0, 60, 70], [50, 60, 70]]))


def test_slice_wrist_cam_kv_rejects_unequal_counts():
    input_ids = torch.tensor([[99, 99, 1], [99, 1, 1]])  # 2 vs 1 patch tokens
    pos = torch.zeros(2, 3, 2)
    try:
        slice_wrist_cam_kv([(pos, pos)], input_ids=input_ids, image_patch_id=99, num_images=1, cam_index=0)
    except ValueError as e:
        assert "unequal image-patch token counts" in str(e)
    else:
        raise AssertionError("expected a ValueError on unequal counts")


def test_gated_depth_read_shape_and_sink_abstains():
    # q, depth K/V in action head space (B, T, H, Dh).
    b, tq, n, h, dh = 2, 4, 6, 3, 8
    q = torch.randn(b, tq, h, dh)
    k_d = torch.randn(b, n, h, dh)
    v_d = torch.randn(b, n, h, dh)

    out = gated_depth_read(q, (k_d, v_d), sink_logit=torch.zeros(h))
    assert out.shape == (b, tq, h, dh)

    # A huge per-head sink logit parks ~all attention mass on the zero-value sink, so the
    # read → 0 (absolute abstaining) regardless of the depth values.
    abstained = gated_depth_read(q, (k_d, v_d), sink_logit=torch.full((h,), 1e4))
    assert torch.allclose(abstained, torch.zeros_like(abstained), atol=1e-5)
    # With a very negative sink logit the sink is ignored and the read is non-trivial.
    admitted = gated_depth_read(q, (k_d, v_d), sink_logit=torch.full((h,), -1e4))
    assert admitted.abs().max() > 1e-3


def test_stream_gradient_flows_to_all_params():
    s = _stream(num_layers=2)
    init = torch.randn(B, N_TOK, 16, requires_grad=True)
    states = s(init, *_wrist_kv(2))
    k, v = s.read_kv(states[-1])
    (k.sum() + v.sum()).backward()
    # Block + read-projection params receive gradient (the gate/sink do not on this
    # path — they live in the action-expert read, exercised by the bit-identity probe).
    assert s.read_k_proj.weight.grad is not None
    assert s.blocks[0].self_attn.q_proj.weight.grad is not None
    assert s.blocks[0].cross_attn.k_proj.weight.grad is not None


def test_encoder_history_slots():
    """Depth history v1 (design §A.6.5): T_h past frames encoded by the same CNN,
    marked by a per-slot time embedding (no re-projection into the current camera
    frame), concatenated oldest → newest ahead of the current frame."""
    cfg = DepthPointmapConfig(
        image_size=(80, 80), patch_size=40, depth_units_mm=1.0, history_num_samples=2
    )
    enc = DepthPointmapEncoder(cfg, d_mem=16).eval()
    n = enc.num_tokens  # 4

    current = torch.full((1, 80, 80), 300.0)
    batch = {
        "observation.depth.wrist": current,
        "history.depth.wrist.depth": torch.stack([current, current], dim=1),  # (1, 2, 80, 80)
        "history.depth.wrist.depth_is_pad": torch.tensor([[False, False]]),
    }
    mem = enc.memory_from_batch(batch, batch_size=1, device=torch.device("cpu"))
    assert mem.shape == (1, 3 * n, 16)
    assert enc.null_memory(2).shape == (2, 3 * n, 16)

    # Identical depth in different slots differs only by the time embedding.
    slot0, slot1, cur = mem[0, :n], mem[0, n : 2 * n], mem[0, 2 * n :]
    assert not torch.allclose(slot0, slot1)
    assert not torch.allclose(slot1, cur)
    assert torch.allclose(slot0 - enc.time_embed[0], slot1 - enc.time_embed[1], atol=1e-5)


def test_encoder_history_pad_and_missing():
    cfg = DepthPointmapConfig(
        image_size=(80, 80), patch_size=40, depth_units_mm=1.0, history_num_samples=2
    )
    enc = DepthPointmapEncoder(cfg, d_mem=16).eval()
    n = enc.num_tokens
    cpu = torch.device("cpu")

    current = torch.full((1, 80, 80), 300.0)
    batch = {
        "observation.depth.wrist": current,
        "history.depth.wrist.depth": torch.stack([current, current], dim=1),
        "history.depth.wrist.depth_is_pad": torch.tensor([[True, False]]),
    }
    mem = enc.memory_from_batch(batch, batch_size=1, device=cpu)
    # Padded oldest slot = null bank + its slot's time embedding.
    assert torch.allclose(mem[0, :n], enc.null_tokens + enc.time_embed[0])
    assert not torch.allclose(mem[0, n : 2 * n], enc.null_tokens + enc.time_embed[1])

    # Missing window: all history slots null, the current slot unaffected.
    mem_missing = enc.memory_from_batch({"observation.depth.wrist": current}, batch_size=1, device=cpu)
    assert mem_missing.shape == (1, 3 * n, 16)
    assert torch.allclose(mem_missing[0, :n], enc.null_tokens + enc.time_embed[0])
    assert torch.allclose(mem_missing[0, n : 2 * n], enc.null_tokens + enc.time_embed[1])
    assert torch.allclose(mem_missing[0, 2 * n :], mem[0, 2 * n :])
