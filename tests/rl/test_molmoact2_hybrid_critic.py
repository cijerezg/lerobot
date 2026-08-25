from types import SimpleNamespace

import pytest
import torch

from lerobot.rl.molmoact2.hybrid_critic import MolmoAct2Critic


def _config(**overrides):
    values = {
        "num_value_bins": 11,
        "critic_llm_depth": 2,
        "critic_input_hidden_size": 32,
        "critic_hidden_size": 16,
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


def test_hybrid_critic_shapes_distribution_and_detached_boundary():
    critic = MolmoAct2Critic(_config())
    encoder_tokens = torch.randn(3, 7, 32, requires_grad=True)
    attention_mask = torch.ones(3, 7, dtype=torch.bool)

    output = critic(encoder_tokens, attention_mask)

    assert output["logits"].shape == (3, 11)
    assert output["probs"].shape == (3, 11)
    assert output["value"].shape == (3, 1)
    torch.testing.assert_close(output["probs"].sum(dim=-1), torch.ones(3))

    output["logits"].sum().backward()
    assert encoder_tokens.grad is None
    assert critic.input_projection.weight.grad is not None


def test_hybrid_critic_ignores_masked_encoder_tokens():
    critic = MolmoAct2Critic(_config()).eval()
    encoder_tokens = torch.randn(2, 7, 32)
    attention_mask = torch.ones(2, 7, dtype=torch.bool)
    attention_mask[:, -2:] = False

    changed_masked_tokens = encoder_tokens.clone()
    changed_masked_tokens[:, -2:] = 1000 * torch.randn_like(changed_masked_tokens[:, -2:])

    with torch.no_grad():
        original = critic(encoder_tokens, attention_mask)["logits"]
        changed = critic(changed_masked_tokens, attention_mask)["logits"]

    torch.testing.assert_close(original, changed, atol=1e-5, rtol=1e-5)


def test_hybrid_critic_distribution_targets_are_normalized():
    critic = MolmoAct2Critic(_config())
    targets = torch.tensor([[-2.0], [-1.0], [0.0]])

    hl_gauss = critic.hl_gauss_target(targets)
    one_hot = critic.one_hot_target(targets)

    assert hl_gauss.shape == (3, 11)
    assert one_hot.shape == (3, 11)
    torch.testing.assert_close(hl_gauss.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(one_hot.sum(dim=-1), torch.ones(3))


def test_hybrid_critic_rejects_sequences_over_configured_limit():
    critic = MolmoAct2Critic(_config(critic_max_tokens=4))
    tokens = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError, match="critic_max_tokens=4"):
        critic(tokens, mask)
