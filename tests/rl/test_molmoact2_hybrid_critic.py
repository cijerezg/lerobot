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


def test_vision_gradients_match_finite_differences_and_preserve_parameter_grads():
    torch.manual_seed(41)
    critic = MolmoAct2Critic(_config(), _backbone()).double().eval()
    ids, images, mask = _prompt(batch=1, seq=5, patches_per_row=1)
    images = images.double()
    # The probe must not zero or accumulate pre-existing trainer gradients.
    param = critic.fusion.distribution_head.weight
    param.grad = torch.ones_like(param)
    before = param.grad.clone()
    with torch.enable_grad():
        out = critic(ids, images, None, mask, vision_grad=True)
    torch.testing.assert_close(param.grad, before)
    assert all(p.grad is None for p in critic.encoder.parameters())
    assert out["vision_grad_norm"].shape == (1,)
    assert out["vision_grad_norm"].item() > 0

    # Perturb each frozen vision feature, not token IDs or trainable weights.
    perturb = torch.zeros(1, HIDDEN, dtype=torch.double)
    hook = critic.encoder.vision_backbone.register_forward_hook(lambda _m, _a, output: output + perturb)
    differences = []
    eps = 1e-5
    try:
        with torch.no_grad():
            for j in range(HIDDEN):
                perturb[0, j] = eps
                plus = critic(ids, images, None, mask)["value"].item()
                perturb[0, j] = -eps
                minus = critic(ids, images, None, mask)["value"].item()
                differences.append((plus - minus) / (2 * eps))
                perturb[0, j] = 0
    finally:
        hook.remove()
    expected = torch.tensor(differences).norm()
    torch.testing.assert_close(out["vision_grad_norm"][0], expected, rtol=0.005, atol=1e-6)


def test_vision_gradient_norm_is_per_frame_and_independent_of_batch_size():
    torch.manual_seed(8)
    critic = MolmoAct2Critic(_config(), _backbone()).eval()
    ids, images, mask = _prompt(batch=2)
    out = critic(ids, images, None, mask, vision_grad=True)
    for i in range(2):
        single = critic(ids[i:i+1], images[i*3:(i+1)*3], None, mask[i:i+1], vision_grad=True)
        torch.testing.assert_close(out["vision_grad_norm"][i], single["vision_grad_norm"][0], rtol=1e-4, atol=1e-6)
    with torch.no_grad():
        normal = critic(ids, images, None, mask)
    torch.testing.assert_close(out["value"], normal["value"], rtol=1e-4, atol=1e-6)
    assert all(p.grad is None for p in critic.parameters())


def test_vision_gradient_rejects_missing_images():
    critic = MolmoAct2Critic(_config(), _backbone()).eval()
    ids, _, mask = _prompt(batch=1)
    with pytest.raises(ValueError, match="visible image patch"):
        critic(ids, None, None, mask, vision_grad=True)


def test_full_input_gradient_covers_text_and_images_and_matches_finite_differences():
    torch.manual_seed(19)
    critic = MolmoAct2Critic(_config(), _backbone()).double().eval()
    ids, images, mask = _prompt(batch=1, seq=4, patches_per_row=1)
    images = images.double()
    param = critic.fusion.distribution_head.weight
    param.grad = torch.ones_like(param)
    before = param.grad.clone()
    out = critic(ids, images, None, mask, input_grad=True, vision_grad=True)
    torch.testing.assert_close(param.grad, before)
    assert all(p.grad is None for p in critic.encoder.parameters())
    assert out["input_token_grad_norms"][0, -1] == 0
    assert out["input_grad_norm"].item() > out["vision_grad_norm"].item()
    perturb = torch.zeros(1, 4, HIDDEN, dtype=torch.double)
    hook = critic.encoder.register_forward_hook(lambda _m, _a, output: output + perturb)
    expected = []
    eps = 1e-5
    try:
        with torch.no_grad():
            for i in range(4):
                components = []
                for j in range(HIDDEN):
                    perturb[0, i, j] = eps
                    plus = critic(ids, images, None, mask)["value"].item()
                    perturb[0, i, j] = -eps
                    minus = critic(ids, images, None, mask)["value"].item()
                    perturb[0, i, j] = 0
                    components.append((plus - minus) / (2 * eps))
                expected.append(torch.tensor(components).norm())
    finally:
        hook.remove()
    expected = torch.stack(expected)
    torch.testing.assert_close(out["input_token_grad_norms"][0], expected, rtol=1e-3, atol=1e-6)
    torch.testing.assert_close(out["input_grad_norm"], expected.norm().reshape(1), rtol=1e-3, atol=1e-6)


def test_full_input_gradient_batch_independence_and_text_only_input():
    torch.manual_seed(7)
    critic = MolmoAct2Critic(_config(), _backbone()).eval()
    ids = torch.tensor([[1, 2, 3, -1], [3, 4, 5, -1]])
    mask = ids != -1
    batched = critic(ids, None, None, mask, input_grad=True)
    assert (batched["input_grad_norm"] > 0).all()
    for i in range(2):
        one = critic(ids[i:i+1], None, None, mask[i:i+1], input_grad=True)
        torch.testing.assert_close(batched["input_grad_norm"][i:i+1], one["input_grad_norm"], rtol=1e-4, atol=1e-6)


STATE_ID = 8
DEPTH_ID = 9


class _DepthPooler(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.wq = nn.Linear(2 * width, width)
        self.wo = nn.Linear(2 * width, width)

    def forward(self, query, inputs_kv, attn_mask=None):
        return self.wq(query) + self.wo(inputs_kv.mean(dim=1, keepdim=True))


class _DepthProjector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.w1 = nn.Linear(width, HIDDEN)

    def forward(self, x):
        return self.w1(x)


def _multimodal_backbone():
    backbone = _backbone()
    vision = backbone.vision_backbone
    width = 8
    vision.vit_config = SimpleNamespace(hidden_size=width)
    vision.image_vit = nn.Module()
    vision.image_vit.transformer = nn.Module()
    vision.image_vit.transformer.resblocks = nn.ModuleList(
        nn.Sequential(nn.Linear(width, width), nn.GELU()) for _ in range(2)
    )
    vision.image_feature_dropout = nn.Dropout(0.1)
    vision.image_pooling_2d = _DepthPooler(width)
    vision.image_projector = _DepthProjector(width)
    # Mimic construction from a frozen actor; the copied depth modules must train.
    vision.requires_grad_(False)
    return backbone


def _multimodal_config(**overrides):
    from lerobot.policies.depth_pointmap.configuration_pointmap import DepthPointmapConfig

    values = dict(
        input_features={"observation.state": SimpleNamespace(shape=(3,))},
        pointmap_config=DepthPointmapConfig(
            image_size=(8, 8), patch_size=2, pooling_size=(2, 2),
            intrinsics=(10.0, 10.0, 4.0, 4.0), depth_units_mm=1.0,
            token_width=8, cnn_hidden_channels=(8, 8), critic_patch_size=2,
            visual_num_blocks=2, visual_feature_taps=(1, 2), visual_source_indices=(0, 1),
        ),
        device="cpu", dtype="float32", gradient_checkpointing=True,
        optimizer_lr=1e-4, critic_lr=1e-3, depth_lr=None,
        critic_target_update_every=1, critic_target_update_weight=1.0,
        task="test", critic_reward_mode="subtask",
    )
    values.update(overrides)
    return _config(**values)


def _multimodal_batch():
    ids = torch.tensor([[1, PATCH_ID, STATE_ID, DEPTH_ID, DEPTH_ID, DEPTH_ID, DEPTH_ID, 2, -1]] * 2)
    return {
        "input_ids": ids,
        "attention_mask": ids != -1,
        "state_values": torch.randn(2, 1, 3),
        "state_values_mask": torch.ones(2, 1, dtype=torch.bool),
        "state_token_id": torch.tensor(STATE_ID),
        "depth_token_id": torch.tensor(DEPTH_ID),
        "observation.depth.wrist": torch.full((2, 1, 8, 8), 200.0),
    }


def _multimodal_forward(critic, batch, **kwargs):
    return critic(batch["input_ids"], None, None, batch["attention_mask"], batch=batch, **kwargs)


def test_continuous_state_changes_values_and_respects_history_mask():
    torch.manual_seed(11)
    critic = MolmoAct2Critic(_multimodal_config(pointmap_config=None), _backbone()).eval()
    batch = {
        "input_ids": torch.tensor([[1, STATE_ID, STATE_ID, -1], [1, STATE_ID, -1, -1]]),
        "state_values": torch.randn(2, 2, 3),
        "state_values_mask": torch.tensor([[True, True], [True, False]]),
        "state_token_id": STATE_ID,
    }
    batch["attention_mask"] = batch["input_ids"] != -1
    original = _multimodal_forward(critic, batch)["value"]
    changed = {**batch, "state_values": batch["state_values"].clone()}
    changed["state_values"][1, 1] += 100
    torch.testing.assert_close(original, _multimodal_forward(critic, changed)["value"])
    changed["state_values"][:, 0] += 2
    assert not torch.allclose(original, _multimodal_forward(critic, changed)["value"])
    original.sum().backward()
    assert critic.fusion.state_projector.weight.grad.norm() > 0
    assert all(p.grad is None for p in critic.encoder.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_depth_and_state_receive_td_gradients_with_frozen_rgb(dtype):
    torch.manual_seed(12)
    backbone = _multimodal_backbone()
    critic = MolmoAct2Critic(_multimodal_config(), backbone)
    critic.fusion.to(dtype=dtype)
    critic.train()
    assert not critic.encoder.training and not critic.encoder.vision_backbone.training
    assert critic.fusion.depth_visual.training
    batch = _multimodal_batch()
    logits = _multimodal_forward(critic, batch)["logits"]
    F.cross_entropy(logits.float(), torch.tensor([2, 7])).backward()
    for module in (
        critic.fusion.state_projector, critic.fusion.pointmap_encoder.cnn,
        critic.fusion.pointmap_encoder.pos_proj, critic.fusion.depth_visual.blocks,
        critic.fusion.depth_visual.pooler, critic.fusion.depth_visual.projector,
    ):
        assert any(p.grad is not None and p.grad.float().norm() > 0 for p in module.parameters())
    assert critic.fusion.depth_marker.grad.float().norm() > 0
    assert all(p.grad is None for p in critic.encoder.parameters())
    assert all(p.grad is None for p in backbone.vision_backbone.parameters())
    source_ids = {id(p) for p in backbone.vision_backbone.parameters()}
    assert source_ids.isdisjoint(id(p) for p in critic.fusion.depth_visual.parameters())


def test_depth_changes_values_and_missing_rows_use_the_learned_null_bank():
    torch.manual_seed(13)
    critic = MolmoAct2Critic(_multimodal_config(), _multimodal_backbone()).eval()
    batch = _multimodal_batch()
    batch["depth.wrist.depth_is_present"] = torch.tensor([True, False])
    with torch.no_grad():
        original = _multimodal_forward(critic, batch)["value"]
        changed = {**batch, "observation.depth.wrist": torch.full((2, 1, 8, 8), 500.0)}
        other = _multimodal_forward(critic, changed)["value"]
        assert not torch.allclose(original[0], other[0])
        torch.testing.assert_close(original[1], other[1])
        missing = {k: v for k, v in batch.items() if k != "observation.depth.wrist"}
        null_output = _multimodal_forward(critic, missing)["value"]
        torch.testing.assert_close(original[1], null_output[1])
        # Depth replaces, rather than adds to, the arbitrary placeholder table row.
        critic.encoder.wte.embedding[DEPTH_ID].add_(1000)
        torch.testing.assert_close(original, _multimodal_forward(critic, batch)["value"])


@pytest.mark.parametrize("modality", ["state", "depth"])
def test_mismatched_continuous_placeholders_fail_instead_of_silently_dropping_inputs(modality):
    critic = MolmoAct2Critic(_multimodal_config(), _multimodal_backbone()).eval()
    batch = _multimodal_batch()
    # Per-row counts must match even when total counts could happen to agree.
    batch["input_ids"][0, 2 if modality == "state" else 3] = 1
    with pytest.raises(ValueError, match="placeholders"):
        _multimodal_forward(critic, batch)


def _policy_with_multimodal_critic():
    from lerobot.rl.molmoact2.rl_molmoact2 import MolmoAct2RLPolicy

    class Policy(nn.Module):
        init_critic = MolmoAct2RLPolicy.init_critic
        load_critic = MolmoAct2RLPolicy.load_critic
        _forward_critic_impl = MolmoAct2RLPolicy._forward_critic_impl
        forward_critic = MolmoAct2RLPolicy.forward_critic
        forward_critic_target = MolmoAct2RLPolicy.forward_critic_target

        def __init__(self):
            super().__init__()
            self.config = _multimodal_config()
            self.backbone = _multimodal_backbone()
            self.backbone.merge_visual_inputs = lambda **kwargs: (None, None)
            self.init_critic()

        def _backbone(self):
            return self.backbone

    return Policy().eval()


def test_policy_forwards_continuous_inputs_and_target_uses_its_own_adapters():
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    torch.manual_seed(14)
    policy = _policy_with_multimodal_critic()
    trainer = MolmoAct2Trainer.__new__(MolmoAct2Trainer)
    batch = _multimodal_batch()
    with torch.no_grad():
        before = policy.forward_critic_target(batch)["value"]
        torch.testing.assert_close(before, policy.forward_critic(batch)["value"])
        policy.critic.fusion.state_projector.weight.add_(2)
        policy.critic.fusion.depth_marker[0].add_(3)
        torch.testing.assert_close(before, policy.forward_critic_target(batch)["value"])
        assert not torch.allclose(before, policy.forward_critic(batch)["value"])
        trainer.update_target_networks(policy)
        torch.testing.assert_close(policy.forward_critic(batch)["value"], policy.forward_critic_target(batch)["value"])
    assert all(not p.requires_grad for p in policy.critic_target.parameters())


def test_input_adapters_are_optimized_and_checkpointed_with_the_critic(tmp_path):
    from safetensors.torch import save_file

    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    policy = _policy_with_multimodal_critic()
    trainer = MolmoAct2Trainer.__new__(MolmoAct2Trainer)
    cfg = SimpleNamespace(policy=policy.config, skip_critic=False)
    trainer._apply_critic_freeze(policy.critic, None, cfg)
    groups = trainer.get_optimizer_groups(policy, cfg)
    critic_group = next(g for g in groups if g["name"] == "critic")
    trainable_ids = {id(p) for p in critic_group["params"]}
    assert trainable_ids == {id(p) for p in policy.critic.fusion.parameters()}
    assert trainable_ids.isdisjoint(id(p) for p in policy.critic_target.parameters())
    # Round-trip real critic.* / critic_target.* keys through the public loader.
    save_file(policy.state_dict(), tmp_path / "model.safetensors")
    restored = _policy_with_multimodal_critic()
    restored.load_critic(str(tmp_path))
    batch = _multimodal_batch()
    with torch.no_grad():
        torch.testing.assert_close(policy.forward_critic(batch)["value"], restored.forward_critic(batch)["value"])
        torch.testing.assert_close(
            policy.forward_critic_target(batch)["value"], restored.forward_critic_target(batch)["value"],
        )


def test_critic_batches_deliver_distinct_current_and_next_depth_to_value_networks():
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    torch.manual_seed(15)
    policy = _policy_with_multimodal_critic()
    trainer = MolmoAct2Trainer.__new__(MolmoAct2Trainer)
    template = _multimodal_batch()
    raw = {
        "state": {"observation.state": torch.zeros(2, 3)},
        "next_state": {"observation.state": torch.ones(2, 3)},
        "reward": torch.zeros(2), "done": torch.zeros(2),
        "complementary_info": {
            "depth.wrist.depth": torch.full((2, 8, 8), 200.0),
            "next_depth.wrist.depth": torch.full((2, 8, 8), 500.0),
        },
    }

    def preprocessor(observations):
        return {
            **template,
            "state_values": observations["observation.state"][:, None],
            "observation.depth.wrist": observations["observation.depth.wrist"],
        }

    current, following, _, _ = trainer._critic_batches(raw, preprocessor, SimpleNamespace(policy=policy.config))
    assert current["observation.depth.wrist"].mean() == 200
    assert following["observation.depth.wrist"].mean() == 500
    assert current["state_values"].mean() == 0
    assert following["state_values"].mean() == 1
    with torch.no_grad():
        assert not torch.allclose(policy.forward_critic(current)["value"], policy.forward_critic_target(following)["value"])
