"""Temporal-PCA objective tests using the actual temporal wrapper and a tiny ViT."""

import copy

import pytest
import torch
from safetensors.torch import load_model, save_model

from lerobot.policies.molmoact2.configuration_molmoact2 import FutureVisualLossConfig
from lerobot.policies.molmoact2.future_visual import (
    FutureVisualObjective,
    normalize_taps,
    temporal_pca,
    weighted_future_l1,
)
from tests.policies.test_molmoact2_mem_encoder import DIM, make_backbone, reference_single_frame


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(41)


def objective():
    backbone = make_backbone()
    backbone.vision_backbone.image_vit.config.hidden_size = DIM
    cfg = FutureVisualLossConfig(
        enabled=True, latent_dim=4, predictor_width=16, predictor_heads=2,
        predictor_layers=1, calibration_pairs=4, calibration_patches_per_camera=4,
        target_update_steps=2, target_encode_batch_size=2,
    )
    return backbone, FutureVisualObjective(backbone.vision_backbone, 2, cfg)


def batch():
    current = torch.randn(4, 2, 4, 6)
    future = torch.randn_like(current)
    valid = torch.tensor([True, True, True, False])
    cameras = torch.tensor([[True, False], [True, True], [False, True], [True, True]])
    return current, future, valid, cameras


def test_target_is_the_plain_vit_with_identical_feature_taps():
    backbone, aux = objective()
    current, future, valid, cameras = batch()
    aux.prepare(backbone.vision_backbone.image_vit, current, future, valid, cameras)
    aux.train()
    assert not aux.target_vit.training
    assert all(not parameter.requires_grad for parameter in aux.target_vit.parameters())
    torch.testing.assert_close(
        aux.encode_target(future),
        reference_single_frame(
            backbone.vision_backbone,
            ((future + 1) * 0.5 * 255).round().clamp(0, 255) / 255 * 2 - 1,
        ),
        rtol=1e-5, atol=1e-6,
    )
    assert bool(aux.ready)
    torch.testing.assert_close(aux.basis.T @ aux.basis, torch.eye(4), atol=2e-5, rtol=2e-5)


def test_auxiliary_gradient_reaches_history_but_never_the_teacher():
    backbone, aux = objective()
    current, future, valid, cameras = batch()
    aux.prepare(backbone.vision_backbone.image_vit, current, future, valid, cameras)
    history = torch.randn(4, 2, 3, 4, 6, requires_grad=True)
    vb = backbone.vision_backbone
    vb._lerobot_history = (history, torch.tensor([3., 2., 1.]), cameras)
    vb._lerobot_capture_future_features = True
    features = vb.encode_image(current)
    assert vb._lerobot_future_features is features
    assert not vb._lerobot_capture_future_features
    loss, metrics = aux(features, future, valid, cameras, torch.zeros(4), "mean")
    loss.backward()
    assert metrics["future_visual_ready"] == 1
    assert history.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in aux.predictor.parameters())
    assert all(p.grad is None for p in aux.target_vit.parameters())


def test_hard_refresh_counts_optimizer_updates_and_resume_preserves_state(tmp_path):
    backbone, aux = objective()
    args = batch()
    source = backbone.vision_backbone.image_vit
    aux.prepare(source, *args)
    old_target = copy.deepcopy(aux.target_vit.state_dict())
    old_basis = aux.basis.clone()
    with torch.no_grad():
        source.patch_embedding.weight.add_(0.2)
    # Additional forwards/accumulation microbatches do not advance the schedule.
    aux.prepare(source, *args)
    aux.prepare(source, *args)
    aux.optimizer_step(source)
    assert int(aux.optimizer_steps) == 1
    for key, value in old_target.items():
        torch.testing.assert_close(aux.target_vit.state_dict()[key], value)
    torch.testing.assert_close(aux.basis, old_basis)
    path = str(tmp_path / "aux.safetensors")
    save_model(aux, path)
    _, resumed = objective()
    load_model(resumed, path, strict=True)
    for key, value in aux.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value)
    resumed.optimizer_step(source)
    assert int(resumed.optimizer_steps) == 2
    assert int(resumed.last_target_update) == 2
    assert int(resumed.last_pca_update) == 2
    for key, value in source.state_dict().items():
        torch.testing.assert_close(resumed.target_vit.state_dict()[key], value)


def test_temporal_pca_rejects_static_background_and_aligns_rotated_coordinates():
    # Static components can have arbitrarily large between-scene variance.
    current = torch.randn(128, 8) * 1000
    delta = torch.zeros_like(current)
    delta[:, 6:] = torch.randn(128, 2)
    basis, _, retained = temporal_pca((current + delta) - current, 2)
    assert retained > 0.999
    assert basis[:6].abs().max() < 1e-5
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    previous = basis @ rotation
    aligned, _, _ = temporal_pca(delta, 2, previous)
    torch.testing.assert_close(aligned, previous, atol=1e-5, rtol=1e-5)
    assert temporal_pca(torch.zeros(128, 8), 2) is None


def test_constant_direction_of_change_is_not_removed_by_centering():
    delta = torch.zeros(32, 8)
    delta[:, 3] = 2
    basis, energy, retained = temporal_pca(delta, 2)
    assert energy == 4
    assert retained == 1
    assert (basis[3].square().sum() - 1).abs() < 1e-6


def test_weighted_l1_excludes_missing_cameras_and_invalid_future_samples():
    prediction = torch.randn(3, 2, 4, 4, requires_grad=True)
    target = torch.randn_like(prediction)
    valid = torch.tensor([True, True, False])
    cameras = torch.tensor([[True, False], [True, True], [True, True]])
    mistake = torch.tensor([0., 1., 1.])
    result = weighted_future_l1(
        prediction, target, valid, cameras, mistake, mistake_weight=1.5, weight=0.2
    )
    norm = lambda x: torch.nn.functional.layer_norm(x, (4,), eps=1e-6)
    per_cam = (norm(prediction) - norm(target)).abs().mean((-1, -2))
    expected = 0.2 * (per_cam[0, 0] + 1.5 * per_cam[1].mean()) / 2.5
    torch.testing.assert_close(result.mean(), expected)
    result.mean().backward()
    assert prediction.grad[0, 1].count_nonzero() == 0
    assert prediction.grad[2].count_nonzero() == 0
    assert prediction.grad[1].abs().sum() > 0


def test_all_invalid_is_finite_zero_and_graph_connected():
    prediction = torch.randn(2, 2, 4, 4, requires_grad=True)
    result = weighted_future_l1(
        prediction, torch.randn_like(prediction), torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, 2, dtype=torch.bool), torch.ones(2),
        mistake_weight=1.5, weight=0.1,
    )
    assert result.tolist() == [0, 0]
    result.mean().backward()
    assert prediction.grad.count_nonzero() == 0


def test_tap_normalization_does_not_let_one_feature_level_dominate():
    features = torch.randn(3, 2, 4, 16)
    expected = normalize_taps(features, 8)
    features[..., :8] = features[..., :8] * 100 + 30
    torch.testing.assert_close(normalize_taps(features, 8), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("kwargs", [
    {"latent_dim": 1}, {"horizon_seconds": 0}, {"target_update_steps": 0},
    {"weight": float("nan")}, {"mistake_weight": 0.5}, {"predictor_width": 7},
])
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        FutureVisualLossConfig(**kwargs)


def test_eval_baselines_do_not_mutate_target_or_pca():
    backbone, aux = objective()
    current, future, valid, cameras = batch()
    aux.prepare(backbone.vision_backbone.image_vit, current, future, valid, cameras)
    snapshot = copy.deepcopy(aux.state_dict())
    aux.eval()
    features = backbone.vision_backbone.encode_image(current)
    _, metrics = aux(
        features, future, valid, cameras, torch.zeros(4), "mean", current_images=current
    )
    assert metrics["future_visual_persistence_loss"] > 0
    assert metrics["future_visual_zero_loss"] > 0
    assert metrics["future_visual_weight_sum"] == 3
    for key, value in snapshot.items():
        torch.testing.assert_close(aux.state_dict()[key], value)


def test_freeze_schedule_and_pretrained_merge_exclude_target_and_protect_predictor():
    from types import SimpleNamespace
    from torch import nn
    from lerobot.rl.molmoact2.rl_molmoact2_trainer import MolmoAct2Trainer

    _, aux = objective()
    policy = nn.Module()
    policy.future_visual = aux
    tp = SimpleNamespace(vision_from_layer=2, language_from_layer=2, depth_warmup=False)
    MolmoAct2Trainer._apply_actor_freeze(policy, tp, freeze_embedding=True)
    assert all(p.requires_grad for p in aux.predictor.parameters())
    assert all(not p.requires_grad for p in aux.target_vit.parameters())
    cfg = SimpleNamespace(policy=SimpleNamespace(optimizer_lr=1e-4, depth_lr=None))
    groups = MolmoAct2Trainer._split_depth_group(
        policy, cfg, [{"name": "policy", "params": list(aux.predictor.parameters()), "lr": 1e-4}]
    )
    protected = next(group for group in groups if group["name"] == "depth")
    assert {id(p) for p in protected["params"]} == {id(p) for p in aux.predictor.parameters()}


def test_real_processor_preserves_camera_order_and_future_is_not_in_policy_inputs():
    from lerobot.types import TransitionKey
    from tests.rl.test_diverse_camera_roles import _pack_step, _transition, ROLE_KEYS

    pack = _pack_step()
    transition = _transition(2, [[True, False, True], [True, True, True]])
    complementary = transition[TransitionKey.COMPLEMENTARY_DATA]
    complementary["future_visual_valid"] = torch.tensor([True, False])
    for index, key in enumerate(ROLE_KEYS):
        complementary[f"future.{key}"] = torch.full((2, 3, 64, 64), 40 * (index + 1), dtype=torch.uint8)
        complementary[f"future.camera_is_present.{key}"] = torch.tensor([True, index != 2])
    clean = copy.deepcopy(transition)
    for key in list(clean[TransitionKey.COMPLEMENTARY_DATA]):
        if key.startswith("future"):
            del clean[TransitionKey.COMPLEMENTARY_DATA][key]
    expected = pack(clean)[TransitionKey.COMPLEMENTARY_DATA]
    actual = pack(transition)[TransitionKey.COMPLEMENTARY_DATA]
    assert actual["future_images"].flatten(0, 1).shape == actual["pixel_values"].shape
    assert actual["future_visual_cameras"].tolist() == [[True, False, True], [True, True, False]]
    assert actual["future_visual_valid"].tolist() == [True, False]
    assert all(not key.startswith("future.") for key in actual)
    for key in ("pixel_values", "input_ids", "attention_mask", "history_images"):
        torch.testing.assert_close(actual[key], expected[key])
    assert not torch.equal(actual["future_images"][:, 0], actual["future_images"][:, 2])


def test_pca_stays_float32_under_autocast():
    delta = torch.randn(64, 8)
    expected = temporal_pca(delta, 4)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = temporal_pca(delta, 4)[0]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_degenerate_refresh_disables_loss_until_calibration_recovers():
    backbone, aux = objective()
    current, future, valid, cameras = batch()
    source = backbone.vision_backbone.image_vit
    aux.prepare(source, current, future, valid, cameras)
    previous = aux.basis.clone()
    aux.calibration_future.copy_(aux.calibration_current)
    aux.optimizer_step(source)
    aux.optimizer_step(source)
    assert not aux.ready
    torch.testing.assert_close(aux.basis, previous)
    features = backbone.vision_backbone.encode_image(current)
    loss, _ = aux(features, future, valid, cameras, torch.zeros(4), "mean")
    assert loss == 0 and torch.isfinite(loss)
    aux.calibration_seen.zero_()
    aux.prepare(source, current, future, valid, cameras)
    assert aux.ready
    assert aux.last_pca_update == 2


def test_checkpoint_key_filter_only_hides_a_fully_absent_auxiliary():
    from types import SimpleNamespace
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    _, aux = objective()
    policy = SimpleNamespace(future_visual=aux)
    keys = [f"future_visual.{key}" for key in aux.state_dict()]
    missing, _ = MolmoAct2Policy._filter_load_keys(policy, keys + ["unrelated"], [])
    assert missing == ["unrelated"]
    partial = ["future_visual.optimizer_steps", "future_visual.predictor.position"]
    missing, _ = MolmoAct2Policy._filter_load_keys(policy, partial, [])
    assert missing == partial
    missing, unexpected = MolmoAct2Policy._filter_load_keys(
        SimpleNamespace(future_visual=None), [], keys + ["unrelated"]
    )
    assert missing == [] and unexpected == ["unrelated"]


def test_fixed_validation_pack_carries_future_targets_and_recorded_mistakes():
    from lerobot.processor.converters import create_transition
    from lerobot.rl.molmoact2.val_loss import ValLoss
    from lerobot.types import TransitionKey
    from tests.rl.test_diverse_camera_roles import _pack_step, ROLE_KEYS

    pack = _pack_step()

    def preprocess(flat):
        transition = create_transition(
            observation={key: value for key, value in flat.items() if str(key).startswith("observation.")},
            action=flat["action"], complementary_data=flat[TransitionKey.COMPLEMENTARY_DATA],
        )
        result = pack(transition)
        return {"action": result[TransitionKey.ACTION], **result[TransitionKey.COMPLEMENTARY_DATA]}

    frames = []
    for row in range(2):
        obs = {"observation.state": torch.zeros(1, 8)}
        future = {}
        for key in (ROLE_KEYS[0], ROLE_KEYS[2]):
            obs[key] = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
            future[key] = torch.full_like(obs[key], 70 + row)
        frames.append({
            "obs": obs, "future_obs": future, "gt_actions": torch.zeros(30, 8),
            "task": "move", "subtask": None, "metadata": None,
            "future_visual_valid": row == 0, "future_visual_mistake": float(row),
        })
    result = ValLoss._pack(frames, preprocess, 30, 8, {}, ROLE_KEYS)
    assert result["future_visual_valid"].tolist() == [True, False]
    assert result["future_visual_mistake"].tolist() == [0., 1.]
    assert result["future_visual_cameras"].tolist() == [[True, False, True], [True, False, True]]
    assert result["future_images"].shape[:2] == (2, 3)
    assert not any(key.startswith("future.") for key in result)
