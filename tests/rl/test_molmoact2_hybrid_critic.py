import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lerobot.rl.molmoact2.hybrid_critic import CriticFusion, MolmoAct2Critic

HIDDEN = 32
PATCH_ID = 7
IMAGE_FEATURE_DIM = 4


def _config(**overrides):
    values = {
        "num_value_bins": 11,
        "critic_llm_depth": 2,
        "critic_num_attention_heads": 4,
        "critic_mlp_ratio": 2.0,
        "critic_dropout": 0.0,
        "critic_max_tokens": 12,
        "value_support_min": -2.0,
        "value_support_max": 0.0,
        "hl_gauss_sigma_ratio": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Embedding(nn.Module):
    """Shape of MolmoAct2Embedding: base table + new-token table."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(16, HIDDEN))
        self.new_embedding = nn.Parameter(torch.randn(4, HIDDEN))

    def forward(self, x):
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(IMAGE_FEATURE_DIM, HIDDEN)

    def forward(self, images, token_pooling):
        return self.proj(images)


def _backbone():
    return SimpleNamespace(
        transformer=SimpleNamespace(wte=_Embedding()),
        vision_backbone=_Vision(),
        config=SimpleNamespace(image_patch_id=PATCH_ID),
    )


def _prompt(batch=2, seq=7, patches_per_row=3):
    input_ids = torch.randint(0, 6, (batch, seq))
    input_ids[:, 1 : 1 + patches_per_row] = PATCH_ID
    input_ids[:, -1] = -1  # padding
    images = torch.randn(batch * patches_per_row, IMAGE_FEATURE_DIM)
    mask = input_ids != -1
    return input_ids, images, mask


def test_fusion_shapes_and_detached_boundary():
    fusion = CriticFusion(_config(), HIDDEN)
    tokens = torch.randn(3, 7, HIDDEN, requires_grad=True)
    mask = torch.ones(3, 7, dtype=torch.bool)

    logits = fusion(tokens, mask)

    assert logits.shape == (3, 11)
    logits.sum().backward()
    assert tokens.grad is None
    assert fusion.value_token.grad is not None
    assert fusion.blocks[0].linear1.weight.grad is not None


def test_fusion_ignores_masked_tokens():
    fusion = CriticFusion(_config(), HIDDEN).eval()
    tokens = torch.randn(2, 7, HIDDEN)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[:, -2:] = False
    changed = tokens.clone()
    changed[:, -2:] = 1000 * torch.randn_like(changed[:, -2:])

    with torch.no_grad():
        torch.testing.assert_close(fusion(tokens, mask), fusion(changed, mask), atol=1e-5, rtol=1e-5)


def test_fusion_rejects_sequences_over_configured_limit():
    fusion = CriticFusion(_config(critic_max_tokens=4), HIDDEN)
    with pytest.raises(ValueError, match="critic_max_tokens=4"):
        fusion(torch.randn(1, 5, HIDDEN), torch.ones(1, 5, dtype=torch.bool))


def test_critic_owns_a_frozen_encoder_copy_and_trains_only_fusion():
    backbone = _backbone()
    critic = MolmoAct2Critic(_config(), backbone)

    assert all(not p.requires_grad for p in critic.encoder.parameters())
    assert all(p.requires_grad for p in critic.fusion.parameters())
    # A copy, not a reference: mutating the backbone leaves the critic untouched.
    with torch.no_grad():
        backbone.transformer.wte.embedding.add_(1.0)
    assert not torch.equal(critic.encoder.wte.embedding, backbone.transformer.wte.embedding)

    input_ids, images, mask = _prompt()
    out = critic(input_ids, images, None, mask)
    assert out["value"].shape == (2, 1)
    assert out["logits"].shape == (2, 11)
    torch.testing.assert_close(out["probs"].sum(dim=-1), torch.ones(2))
    out["logits"].sum().backward()
    assert critic.fusion.distribution_head.weight.grad is not None
    assert all(p.grad is None for p in critic.encoder.parameters())

    keys = set(critic.state_dict())
    assert "encoder.wte.embedding" in keys and "encoder.vision_backbone.proj.weight" in keys
    assert "fusion.blocks.0.linear1.weight" in keys and "fusion.value_token" in keys


def test_encoder_adds_image_features_on_patch_positions_only():
    critic = MolmoAct2Critic(_config(), _backbone())
    input_ids, images, _ = _prompt(batch=1, seq=6, patches_per_row=2)
    input_ids[:, -1] = 3  # no padding here, keep -1 handling to its own test

    with torch.no_grad():
        text_only = critic.encoder(input_ids, None, None)
        fused = critic.encoder(input_ids, images, None)
        expected_features = critic.encoder.vision_backbone(images, None)

    is_patch = input_ids[0] == PATCH_ID
    torch.testing.assert_close(fused[0, ~is_patch], text_only[0, ~is_patch])
    torch.testing.assert_close(fused[0, is_patch] - text_only[0, is_patch], expected_features)


def test_encoder_maps_padding_ids_to_row_zero():
    critic = MolmoAct2Critic(_config(), _backbone())
    ids = torch.tensor([[-1, 2]])
    with torch.no_grad():
        tokens = critic.encoder(ids, None, None)
    torch.testing.assert_close(tokens[0, 0], critic.encoder.wte.embedding[0])


def test_target_fusion_runs_on_the_shared_encoder():
    critic = MolmoAct2Critic(_config(), _backbone()).eval()
    target = copy.deepcopy(critic.fusion).eval()
    input_ids, images, mask = _prompt()

    with torch.no_grad():
        same = critic(input_ids, images, None, mask, fusion=target)["logits"]
        live = critic(input_ids, images, None, mask)["logits"]
        target.distribution_head.bias.add_(1.0)
        shifted = critic(input_ids, images, None, mask, fusion=target)["logits"]

    torch.testing.assert_close(same, live)
    torch.testing.assert_close(shifted, live + 1.0)


def test_critic_distribution_targets_are_normalized():
    critic = MolmoAct2Critic(_config(), _backbone())
    targets = torch.tensor([[-2.0], [-1.0], [0.0]])

    hl_gauss = critic.hl_gauss_target(targets)
    one_hot = critic.one_hot_target(targets)

    assert hl_gauss.shape == (3, 11) and one_hot.shape == (3, 11)
    torch.testing.assert_close(hl_gauss.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(one_hot.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(critic.value_from_probs(one_hot), targets)
