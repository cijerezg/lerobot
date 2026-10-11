"""Separate hand parameters (policies/molmoact2/hand_head.py): the expert's hand head and the
hand framing tokens' norm-pinned rows, and how the trainer treats them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.molmoact2.configuration_molmoact2 import HandBlockConfig
from lerobot.policies.molmoact2.hand_block import HAND_ACTION_SLOTS
from lerobot.policies.molmoact2.hand_head import (
    HandExpertHead,
    HandTokenRows,
    install_hand_expert_head,
    install_hand_token_rows,
)


class _Expert(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.action_embed = nn.Linear(32, width)
        self.final_layer = nn.Module()
        self.final_layer.linear = nn.Linear(width, 32)


def test_hand_expert_head_equals_the_single_projection_at_init_and_splits_the_gradients():
    torch.manual_seed(0)
    expert = _Expert()
    x = torch.randn(2, 5, 32)
    h = torch.randn(2, 5, 16)
    ref_embed = expert.action_embed(x).detach().clone()
    ref_out = expert.final_layer.linear(h).detach().clone()

    head = install_hand_expert_head(expert)
    assert isinstance(head, HandExpertHead)
    assert install_hand_expert_head(expert) is head  # idempotent: no second set of hooks
    names = dict(expert.named_parameters())
    assert "hand_head.hand_in.weight" in names and "hand_head.hand_out.weight" in names

    embed = expert.action_embed(x)
    out = expert.final_layer.linear(h)
    # Input side: hand_in starts as the copied columns, so the sum is the original projection.
    assert torch.allclose(embed, ref_embed, atol=1e-5)
    # Output side: the checkpoint's slots outside the hand block, hand_out (zero) inside it.
    assert torch.equal(out[..., : HAND_ACTION_SLOTS.start], ref_out[..., : HAND_ACTION_SLOTS.start])
    assert torch.equal(out[..., HAND_ACTION_SLOTS.stop :], ref_out[..., HAND_ACTION_SLOTS.stop :])
    assert torch.equal(out[..., HAND_ACTION_SLOTS], torch.zeros_like(out[..., HAND_ACTION_SLOTS]))

    (embed.square().sum() + out.square().sum() + out.sum()).backward()
    # The original matrices get no gradient on the hand slots; the fresh ones do.
    assert torch.equal(
        expert.action_embed.weight.grad[:, HAND_ACTION_SLOTS],
        torch.zeros_like(expert.action_embed.weight.grad[:, HAND_ACTION_SLOTS]),
    )
    assert bool((expert.action_embed.weight.grad[:, : HAND_ACTION_SLOTS.start] != 0).any())
    assert torch.equal(
        expert.final_layer.linear.weight.grad[HAND_ACTION_SLOTS],
        torch.zeros_like(expert.final_layer.linear.weight.grad[HAND_ACTION_SLOTS]),
    )
    assert torch.equal(
        expert.final_layer.linear.bias.grad[HAND_ACTION_SLOTS],
        torch.zeros_like(expert.final_layer.linear.bias.grad[HAND_ACTION_SLOTS]),
    )
    assert bool((head.hand_in.weight.grad != 0).any())
    assert bool((head.hand_out.weight.grad != 0).any())  # zero weight, nonzero dL/dy . h^T
    assert bool((head.hand_out.bias.grad != 0).any())

    # Detaching the hooks gives the checkpoint's modules back unchanged.
    head.detach_hooks()
    assert torch.equal(expert.final_layer.linear(h), ref_out)


def test_hand_expert_head_runs_in_the_expert_dtype():
    expert = _Expert().to(torch.bfloat16)
    head = install_hand_expert_head(expert)
    assert head.hand_in.weight.dtype == torch.bfloat16
    out = expert.final_layer.linear(torch.randn(3, 16, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16 and out.shape == (3, 32)
    assert expert.action_embed(torch.randn(3, 32, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_hand_expert_head_refuses_an_expert_narrower_than_the_hand_block():
    expert = _Expert()
    expert.action_embed = nn.Linear(8, 16)
    with pytest.raises(RuntimeError, match="action_embed"):
        install_hand_expert_head(expert)


def test_hand_token_rows_start_as_the_seeds_pin_the_norm_and_swap_only_the_hand_ids():
    torch.manual_seed(0)
    wte = nn.Embedding(20, 6)
    start_id, end_id = 10, 11
    table = wte.weight.detach().clone()
    rows = install_hand_token_rows(wte, (start_id, end_id))
    assert isinstance(rows, HandTokenRows)
    assert torch.allclose(rows.rows.detach(), table[[start_id, end_id]])
    assert rows.cosines() == {
        "hand_token_cos_start": pytest.approx(1.0),
        "hand_token_cos_end": pytest.approx(1.0),
    }

    # Move the free parameter: the effective rows keep the seeds' norms, direction only changes.
    with torch.no_grad():
        rows.rows.add_(torch.randn_like(rows.rows) * 3.0)
    effective = rows.effective()
    assert torch.allclose(effective.norm(dim=-1), table[[start_id, end_id]].norm(dim=-1), atol=1e-5)
    assert rows.cosines()["hand_token_cos_start"] < 0.99

    ids = torch.tensor([[1, start_id, end_id, 3]])
    out = wte(ids)
    assert torch.allclose(out[0, 1], effective[0]) and torch.allclose(out[0, 2], effective[1])
    assert torch.equal(out[0, 0], table[1]) and torch.equal(out[0, 3], table[3])

    # Gradient reaches the parameter and is tangent to the sphere (the radial part is dropped).
    out[0, 1].sum().backward()
    grad = rows.rows.grad
    assert grad is not None and bool((grad[0] != 0).any())
    assert torch.equal(grad[1], torch.zeros_like(grad[1]))  # <hand_end> not in the loss here
    assert abs(float(torch.dot(grad[0], rows.rows.detach()[0]))) < 1e-4 * float(
        grad[0].norm() * rows.rows.detach()[0].norm()
    )

    # Renormalization: back on the sphere, same function.
    before = rows.effective().detach().clone()
    rows.renormalize()
    assert torch.allclose(rows.rows.detach().norm(dim=-1), rows.norm, atol=1e-5)
    assert torch.allclose(rows.effective().detach(), before, atol=1e-5)

    # Lookups that are not (ids -> rows) pass through, e.g. a scalar marker lookup.
    scalar = wte(torch.tensor(start_id))
    assert scalar.shape == (6,) and torch.allclose(scalar, effective[0])
    rows.detach_hook()
    assert torch.equal(wte(ids)[0, 1], table[start_id])


def test_hand_block_config_validates_token_lr():
    HandBlockConfig(token_lr=None)
    HandBlockConfig(token_lr=1e-3)
    with pytest.raises(ValueError, match="token_lr"):
        HandBlockConfig(token_lr=0.0)


def test_filter_load_keys_initializes_absent_hand_parameters_afresh():
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    model = SimpleNamespace(future_visual=None)
    missing = [
        "model.model.action_expert.hand_head.hand_in.weight",
        "hand_token_rows.rows",
        "hand_token_rows.norm",
        "model.lm_head.weight",
    ]
    unexpected = ["model.model.action_expert.hand_head.hand_out.bias", "other"]
    missing, unexpected = MolmoAct2Policy._filter_load_keys(model, missing, unexpected)
    assert missing == ["model.lm_head.weight"]
    assert unexpected == ["other"]


class _Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.wte = nn.Embedding(20, 6)
        self.model.lm_head = nn.Linear(6, 20)
        self.model.action_expert = _Expert()
        install_hand_expert_head(self.model.action_expert)
        self.hand_token_rows = install_hand_token_rows(self.model.transformer.wte, (10, 11))
        self.pointmap_encoder = nn.Linear(2, 2)


def test_trainer_keeps_the_hand_parameters_trainable_and_in_the_fresh_group():
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    policy = _Policy()
    tp = SimpleNamespace(vision_from_layer=2, language_from_layer=2, depth_warmup=False)
    MolmoAct2Trainer._apply_actor_freeze(policy, tp, freeze_embedding=True)
    trainable = {name for name, p in policy.named_parameters() if p.requires_grad}
    assert "hand_token_rows.rows" in trainable
    assert "model.action_expert.hand_head.hand_in.weight" in trainable
    assert "model.action_expert.hand_head.hand_out.weight" in trainable
    assert "model.transformer.wte.weight" not in trainable

    cfg = SimpleNamespace(policy=SimpleNamespace(optimizer_lr=3e-5, depth_lr=None, hand=HandBlockConfig()))
    groups = MolmoAct2Trainer._split_depth_group(
        policy, cfg, [{"name": "policy", "params": list(policy.parameters()), "lr": 3e-5}]
    )
    by_name = {g["name"]: g for g in groups}
    depth_ids = {id(p) for p in by_name["depth"]["params"]}
    policy_ids = {id(p) for p in by_name["policy"]["params"]}
    fresh = [p for name, p in policy.named_parameters() if "hand_head" in name or "hand_token_rows" in name]
    assert fresh and all(id(p) in depth_ids for p in fresh)
    assert all(id(p) not in policy_ids for p in fresh)
    assert id(policy.model.action_expert.action_embed.weight) in policy_ids


def test_trainer_gives_the_hand_token_rows_their_own_lr_inside_the_depth_optimizer():
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer
    from lerobot.rl.utils import build_named_adamw_optimizers

    policy = _Policy()
    cfg = SimpleNamespace(
        policy=SimpleNamespace(optimizer_lr=3e-5, depth_lr=None, hand=HandBlockConfig(token_lr=1e-3))
    )
    groups = MolmoAct2Trainer._split_depth_group(
        policy, cfg, [{"name": "policy", "params": list(policy.parameters()), "lr": 3e-5}]
    )
    depth = next(g for g in groups if g["name"] == "depth")
    assert isinstance(depth["params"][0], dict) and len(depth["params"]) == 2
    assert depth["params"][1]["lr"] == 1e-3
    assert [id(p) for p in depth["params"][1]["params"]] == [id(policy.hand_token_rows.rows)]

    optimizers = build_named_adamw_optimizers(groups, SimpleNamespace(), seed=0)
    lrs = [group["lr"] for group in optimizers["depth"].param_groups]
    assert lrs == [3e-5, 1e-3]


def test_ar_decode_accepts_added_markers_and_uses_learned_hand_rows():
    from lerobot.policies.molmoact2.modeling_molmoact2 import _patch_ar_decode_embeddings

    class SplitEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Parameter(torch.randn(10, 4))
            self.new_embedding = nn.Parameter(torch.randn(2, 4))

        def forward(self, ids):
            return nn.functional.embedding(ids, torch.cat([self.embedding, self.new_embedding]))

    wte = SplitEmbedding()
    model = SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(wte=wte)))
    model._embed_base_tokens = lambda ids: nn.functional.embedding(ids, wte.embedding)
    rows = install_hand_token_rows(wte, (10, 11))
    with torch.no_grad():
        rows.rows.copy_(rows.rows.roll(1, dims=1))
    tokens = torch.tensor([[2, 10, 11]])
    with pytest.raises(IndexError):
        model._embed_base_tokens(tokens)
    _patch_ar_decode_embeddings(model)
    _patch_ar_decode_embeddings(model)
    decoded = model._embed_base_tokens(tokens)
    torch.testing.assert_close(decoded[:, 0], wte.embedding[2:3])
    torch.testing.assert_close(decoded[0, 1:], rows.effective())
    assert not torch.allclose(decoded[0, 1:], wte.new_embedding)
