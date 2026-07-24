"""MEM video encoder (04_memory.md §2.4): temporal attention, e(t), drop, invariants.

Uses stub modules mirroring the remote-code ViT interface (attention with
wq/wk/wv/wo + norms + mlp) so no checkpoint load is needed. The real-checkpoint
behavior is covered by the smoke script.
"""

import math

import pytest

pytest.importorskip("transformers", reason="molmoact2 modeling module imports policy deps")

import torch
import torch.nn.functional as F
from torch import nn

from lerobot.policies.molmoact2.modeling_molmoact2 import (  # noqa: E402
    _patch_memory_efficient_vision_backbone,
    _sinusoidal_seconds_embedding,
    _temporal_vision_block,
)

torch.manual_seed(0)

DIM, HEADS, PATCHES = 8, 2, 4


class StubAttention(nn.Module):
    """Mirror of ViTMultiHeadDotProductAttention's eager path (float32 attention)."""

    def __init__(self):
        super().__init__()
        self.num_heads = HEADS
        self.num_key_value_heads = HEADS
        self.head_dim = DIM // HEADS
        self.num_key_value_groups = 1
        self.float32_attention = True
        self.attention_dropout = 0.0
        self.wq = nn.Linear(DIM, DIM)
        self.wk = nn.Linear(DIM, DIM)
        self.wv = nn.Linear(DIM, DIM)
        self.wo = nn.Linear(DIM, DIM)
        self.residual_dropout = nn.Dropout(0.0)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.wq(x).view(b, n, HEADS, self.head_dim).to(torch.float32)
        k = self.wk(x).view(b, n, HEADS, self.head_dim).to(torch.float32)
        v = self.wv(x).view(b, n, HEADS, self.head_dim)
        w = torch.einsum("bqhd,bkhd->bhqk", q / math.sqrt(self.head_dim), k)
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.einsum("bhqk,bkhd->bqhd", w, v).reshape(b, n, DIM)
        return self.residual_dropout(self.wo(out))


class StubBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = StubAttention()
        self.attention_norm = nn.LayerNorm(DIM)
        self.feed_forward = nn.Sequential(nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, DIM))
        self.ffn_norm = nn.LayerNorm(DIM)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


def naive_union_block(block, x, e_t, history_on):
    """Per-query loops: for patch i of frame t, one softmax over its own frame's n
    patches + patch i of strictly older frames. The reference for the vectorized op."""
    attn = block.attention
    x = x + e_t[None, :, None, :]
    h = block.attention_norm(x)
    bc, t_total, n, _ = h.shape
    hd = attn.head_dim
    q = attn.wq(h).view(bc, t_total, n, HEADS, hd).to(torch.float32)
    k = attn.wk(h).view(bc, t_total, n, HEADS, hd).to(torch.float32)
    v = attn.wv(h).view(bc, t_total, n, HEADS, hd)
    out = torch.zeros_like(v)
    for b in range(bc):
        for t in range(t_total):
            for i in range(n):
                for head in range(HEADS):
                    keys = [k[b, t, j, head] for j in range(n)]
                    vals = [v[b, t, j, head] for j in range(n)]
                    if history_on is None or history_on[b]:
                        keys += [k[b, s, i, head] for s in range(t)]
                        vals += [v[b, s, i, head] for s in range(t)]
                    scores = torch.stack(
                        [q[b, t, i, head] @ key / math.sqrt(hd) for key in keys]
                    )
                    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
                    out[b, t, i, head] = sum(w * val for w, val in zip(weights, vals, strict=True))
    out = out.reshape(bc, t_total, n, DIM)
    x = x + attn.residual_dropout(attn.wo(out))
    return x + block.feed_forward(block.ffn_norm(x))


def test_e_t_boundary_condition_and_extrapolation():
    e = _sinusoidal_seconds_embedding(torch.tensor([0.0, 1.0, 5.0, 54.0]), DIM)
    assert torch.equal(e[0], torch.zeros(DIM))  # e(0) = 0 exactly
    assert e[1:].abs().sum() > 0
    assert torch.isfinite(e).all()


def test_temporal_block_matches_naive_reference():
    block = StubBlock().eval()
    x = torch.randn(2, 3, PATCHES, DIM)
    e_t = _sinusoidal_seconds_embedding(torch.tensor([2.0, 1.0, 0.0]), DIM)
    got = _temporal_vision_block(block, x, e_t, None)
    want = naive_union_block(block, x, e_t, None)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()


def test_temporal_block_masked_sample_is_spatial_only():
    """history_on=False must reproduce the plain block exactly on the current frame
    (the K=1 pretrained op): masked temporal keys get exp(-inf)=0 mass."""
    block = StubBlock().eval()
    x = torch.randn(2, 3, PATCHES, DIM)
    e_t = _sinusoidal_seconds_embedding(torch.tensor([2.0, 1.0, 0.0]), DIM)
    history_on = torch.tensor([False, True])
    got = _temporal_vision_block(block, x, e_t, history_on)
    # Sample 0's current frame: e(0)=0 and no temporal keys → plain spatial block.
    want = block(x[0, -1][None])
    assert torch.allclose(got[0, -1][None], want, atol=1e-6), (got[0, -1] - want).abs().max()
    # Sample 1 (history on) must differ from its spatial-only counterpart.
    assert not torch.allclose(got[1, -1][None], block(x[1, -1][None]), atol=1e-4)
    # And equal the naive reference under the same mask.
    assert torch.allclose(got, naive_union_block(block, x, e_t, history_on), atol=1e-5)


def test_temporal_block_causality():
    """Perturbing frame s changes only frames >= s (strictly older frames are keys
    for newer queries, never the reverse)."""
    block = StubBlock().eval()
    x = torch.randn(1, 3, PATCHES, DIM)
    e_t = _sinusoidal_seconds_embedding(torch.tensor([2.0, 1.0, 0.0]), DIM)
    base = _temporal_vision_block(block, x, e_t, None)
    perturbed = x.clone()
    # Random noise, not a constant: a uniform shift is removed by the pre-attention
    # LayerNorm and would never reach the temporal keys.
    perturbed[:, 1] += torch.randn_like(perturbed[:, 1])
    out = _temporal_vision_block(block, perturbed, e_t, None)
    assert torch.equal(out[:, 0], base[:, 0])  # older frame untouched
    assert not torch.allclose(out[:, 1], base[:, 1])
    assert not torch.allclose(out[:, 2], base[:, 2])  # current reads the past


# ── encode_image: stash, drop, shape invariant, bit-identity ─────────────────


class StubViT(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.patch_embedding = nn.Linear(6, DIM)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([StubBlock() for _ in range(num_layers)])
        self.config = type("C", (), {"image_num_patch": (2, 2)})()

    def add_pos_emb(self, x, patch_num):
        return x


class StubVisionBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_vit = StubViT()
        # The remote code normalizes negative taps to positive at init ([-3, -9] →
        # [24, 18] on the 27-layer ViT); mirror that: taps at 6 and 2 of 8 layers.
        self.vit_layers = [6, 2]
        self.num_prefix_tokens = 0


def make_backbone():
    backbone = nn.Module()
    backbone.vision_backbone = StubVisionBackbone().eval()
    _patch_memory_efficient_vision_backbone(
        backbone, gradient_checkpointing=False, temporal_layer_stride=4
    )
    return backbone


def reference_single_frame(vb, images):
    """The pre-MEM encoder loop, straight-line."""
    b, c, n, p = images.shape
    x = vb.image_vit.patch_embedding(images.view(b * c, n, p))
    feats = {}
    for idx, block in enumerate(vb.image_vit.transformer.resblocks):
        x = block(x)
        feats[idx] = x
    out = torch.cat([feats[int(layer)] for layer in vb.vit_layers], dim=-1)
    return out.view(b, c, n, -1)


def test_encode_image_no_history_bit_identical():
    backbone = make_backbone()
    vb = backbone.vision_backbone
    images = torch.randn(2, 2, PATCHES, 6)
    assert torch.equal(vb.encode_image(images), reference_single_frame(vb, images))


def test_encode_image_history_keeps_output_shape_and_consumes_stash():
    backbone = make_backbone()
    vb = backbone.vision_backbone
    images = torch.randn(2, 2, PATCHES, 6)
    frames = torch.randn(2, 2, 3, PATCHES, 6)  # (B, cams, T_h, n, patch_dim)
    times = torch.tensor([3.0, 2.0, 1.0])
    mask = torch.tensor([True, True])

    vb._lerobot_history = (frames, times, mask)
    out = vb.encode_image(images)
    assert out.shape == reference_single_frame(vb, images).shape  # token-count invariant
    assert vb._lerobot_history is None  # consume-once
    # History must actually change the features.
    assert not torch.allclose(out, reference_single_frame(vb, images), atol=1e-4)


def test_encode_image_all_masked_equals_no_history():
    """history_dropout for every sample → exact K=1 features despite the stash."""
    backbone = make_backbone()
    vb = backbone.vision_backbone
    images = torch.randn(2, 2, PATCHES, 6)
    frames = torch.randn(2, 2, 3, PATCHES, 6)
    vb._lerobot_history = (frames, torch.tensor([3.0, 2.0, 1.0]), torch.tensor([False, False]))
    out = vb.encode_image(images)
    assert torch.allclose(out, reference_single_frame(vb, images), atol=1e-5)


def test_encode_image_camera_count_mismatch_raises():
    backbone = make_backbone()
    vb = backbone.vision_backbone
    vb._lerobot_history = (torch.randn(2, 1, 3, PATCHES, 6), torch.tensor([3.0, 2.0, 1.0]), None)
    with pytest.raises(ValueError, match="one crop per camera"):
        vb.encode_image(torch.randn(2, 2, PATCHES, 6))
