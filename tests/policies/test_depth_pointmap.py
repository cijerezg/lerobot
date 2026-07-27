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


# --- MoT co-evolving depth stream (depth_pointmap_design.md Part B) ----------------

from lerobot.policies.depth_pointmap.modeling_stream import (  # noqa: E402
    DepthStream,
    depth_attention_mass,
    join_depth_columns,
    mask_camera_patch_span,
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


def test_stream_depth_bias_init():
    # Joint softmax (depth_redesign_options.md §5.2): the α gate and sink are gone;
    # the per-layer depth-column bias starts at −2 (soft start, not a hard zero).
    s = _stream(num_layers=4, heads=2)
    assert not hasattr(s, "gate") and not hasattr(s, "sink_logit")
    assert s.depth_bias.shape == (4,)
    assert torch.equal(s.depth_bias, torch.full((4,), -2.0))


def test_stream_read_kv_into_action_head_space():
    s = _stream(num_layers=2, heads=3, head_dim=8)
    init = torch.randn(B, N_TOK, 16)
    state = s(init, *_wrist_kv(2))[0]
    k, v = s.read_kv(state)
    assert k.shape == (B, N_TOK, 3, 8)
    assert v.shape == (B, N_TOK, 3, 8)


def test_stream_cross_on_kills_wrist_bridge():
    """RGB dropout masks the wrist span out of attention, but the stream's K/V
    gather bypasses attention masks — cross_on=False must make the row's output
    independent of the wrist content (and gradient-dead to it)."""
    torch.manual_seed(0)
    s = _stream(num_layers=2).eval()
    init = torch.randn(B, N_TOK, 16)
    keys_a, values_a = _wrist_kv(2)
    # Same wrist content for row 0, garbage for row 1.
    keys_b = [k.clone() for k in keys_a]
    values_b = [v.clone() for v in values_a]
    for k, v in zip(keys_b, values_b, strict=True):
        k[1] = torch.randn_like(k[1]) * 100
        v[1] = torch.randn_like(v[1]) * 100
    cross_on = torch.tensor([True, False])
    out_a = s(init, keys_a, values_a, cross_on=cross_on)[-1]
    out_b = s(init, keys_b, values_b, cross_on=cross_on)[-1]
    assert torch.allclose(out_a[1], out_b[1])  # bridge-killed row ignores wrist content
    assert torch.allclose(out_a[0], out_b[0])  # row 0 (bridge on, same content) unchanged
    out_on = s(init, keys_a, values_a, cross_on=torch.tensor([True, True]))[-1]
    assert not torch.allclose(out_on[1], out_a[1])  # bridge actually matters when on


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


def test_rgb_dropout_masks_only_that_camera_span():
    """The processor's RGB-dropout edit: zero the dropped camera's <im_patch>
    columns of attention_mask for dropped rows only, leaving every other token
    (text, the other camera, padding state) untouched."""
    pid = 99
    # 2 cams × 3 patch tokens, text on both sides; row 1 is the dropped sample.
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


def test_slice_wrist_cam_kv_rejects_unequal_counts():
    input_ids = torch.tensor([[99, 99, 1], [99, 1, 1]])  # 2 vs 1 patch tokens
    pos = torch.zeros(2, 3, 2)
    try:
        slice_wrist_cam_kv([(pos, pos)], input_ids=input_ids, image_patch_id=99, num_images=1, cam_index=0)
    except ValueError as e:
        assert "unequal image-patch token counts" in str(e)
    else:
        raise AssertionError("expected a ValueError on unequal counts")


def _joint_inputs(b=2, tq=4, t_ctx=7, n=6, h=3, dh=8, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, tq, h, dh)
    k = torch.randn(b, t_ctx, h, dh)
    v = torch.randn(b, t_ctx, h, dh)
    k_d = torch.randn(b, n, h, dh)
    v_d = torch.randn(b, n, h, dh)
    ctx_mask = torch.zeros(b, 1, 1, t_ctx)
    ctx_mask[0, ..., -1] = torch.finfo(torch.float32).min  # one padded context column
    return q, k, v, k_d, v_d, ctx_mask


def test_join_depth_columns_mask_and_shapes():
    q, k, v, k_d, v_d, ctx_mask = _joint_inputs()
    bias = torch.tensor(-2.0)
    k_j, v_j, mask_j = join_depth_columns(ctx_mask, k=k, v=v, depth_kv=(k_d, v_d), depth_bias=bias)
    assert k_j.shape == (2, 7 + 6, 3, 8) and v_j.shape == k_j.shape
    assert mask_j.shape == (2, 1, 1, 13)
    assert torch.equal(mask_j[..., :7], ctx_mask)  # context columns untouched (incl. padding)
    assert torch.all(mask_j[..., 7:] == -2.0)  # depth columns carry b_ℓ
    # None context mask ⇒ zeros for context columns.
    _, _, mask_none = join_depth_columns(None, k=k, v=v, depth_kv=(k_d, v_d), depth_bias=bias)
    assert torch.all(mask_none[..., :7] == 0) and torch.all(mask_none[..., 7:] == -2.0)


def test_joint_softmax_matches_eager_and_bias_gets_gradient():
    """The b_ℓ gradient flows through SDPA's attn_mask — the one backend-dependent
    assumption of the joint read. Verified against an eager softmax reference for
    both the output and the gradients (bias, depth values)."""
    q, k, v, k_d, v_d, ctx_mask = _joint_inputs()
    bias_sdpa = torch.nn.Parameter(torch.tensor(-2.0))
    k_d_sdpa = k_d.clone().requires_grad_(True)
    v_d_sdpa = v_d.clone().requires_grad_(True)
    k_j, v_j, mask_j = join_depth_columns(
        ctx_mask, k=k, v=v, depth_kv=(k_d_sdpa, v_d_sdpa), depth_bias=bias_sdpa
    )
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k_j.transpose(1, 2), v_j.transpose(1, 2), attn_mask=mask_j
    ).transpose(1, 2)
    out_sdpa.pow(2).sum().backward()

    bias_ref = torch.nn.Parameter(torch.tensor(-2.0))
    k_d_ref = k_d.clone().requires_grad_(True)
    v_d_ref = v_d.clone().requires_grad_(True)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, torch.cat([k, k_d_ref], dim=1)) / (8**0.5)
    scores = scores + torch.cat(
        [ctx_mask, bias_ref.reshape(1, 1, 1, 1).expand(2, 1, 1, 6)], dim=-1
    )
    weights = scores.softmax(dim=-1)
    out_ref = torch.einsum("bhqk,bkhd->bqhd", weights, torch.cat([v, v_d_ref], dim=1))
    out_ref.pow(2).sum().backward()

    assert torch.allclose(out_sdpa, out_ref, atol=1e-5)
    assert bias_sdpa.grad is not None and bias_sdpa.grad.abs() > 1e-6
    assert torch.allclose(bias_sdpa.grad, bias_ref.grad, atol=1e-4)
    assert torch.allclose(k_d_sdpa.grad, k_d_ref.grad, atol=1e-4)
    assert torch.allclose(v_d_sdpa.grad, v_d_ref.grad, atol=1e-4)


def test_depth_attention_mass_extremes():
    q, k, v, k_d, v_d, ctx_mask = _joint_inputs()
    k_j, _, mask_open = join_depth_columns(ctx_mask, k=k, v=v, depth_kv=(k_d, v_d), depth_bias=torch.tensor(1e4))
    assert depth_attention_mass(q, k_j, mask_open, num_depth=6) > 0.999  # bias ≫ ⇒ all mass on depth
    _, _, mask_shut = join_depth_columns(ctx_mask, k=k, v=v, depth_kv=(k_d, v_d), depth_bias=torch.tensor(-1e4))
    assert depth_attention_mass(q, k_j, mask_shut, num_depth=6) < 1e-3  # bias ≪ ⇒ depth ignored


def test_stream_gradient_flows_to_all_params():
    s = _stream(num_layers=2)
    init = torch.randn(B, N_TOK, 16, requires_grad=True)
    states = s(init, *_wrist_kv(2))
    k, v = s.read_kv(states[-1])
    (k.sum() + v.sum()).backward()
    # Block + read-projection params receive gradient (depth_bias does not on this
    # path — it enters through the joint-softmax mask, exercised above).
    assert s.read_k_proj.weight.grad is not None
    assert s.blocks[0].self_attn.q_proj.weight.grad is not None
    assert s.blocks[0].cross_attn.k_proj.weight.grad is not None


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


def test_config_rejects_negative_history_samples():
    with pytest.raises(ValueError, match="history_num_samples"):
        DepthPointmapConfig(history_num_samples=-1)
    with pytest.raises(ValueError, match="history_num_samples"):
        MemoryConfig(history_num_samples=-1)
    with pytest.raises(ValueError, match="history_num_samples"):
        MemoryConfig(history_keys=["depth.wrist.depth"], history_num_samples=0)
