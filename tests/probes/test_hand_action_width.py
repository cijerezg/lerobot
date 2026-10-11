"""Hand-augmented inference decodes all slots but motion comparisons use joint slots."""
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.probes.adapters.molmoact2 import MolmoAct2Adapter
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.mark.parametrize('mode', ['continuous', 'discrete'])
@pytest.mark.parametrize('batched', [False, True])
def test_adapter_decodes_full_hand_chunk_before_joint_metrics(mode, batched):
    config = MolmoAct2Config(
        hand_block=True,
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(8,))},
    )
    config.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(8,))
    config.validate_features()
    config.action_encoding = 'anchor'
    calls = []
    policy = SimpleNamespace(config=config)

    def predict(batch, **kwargs):
        width = MolmoAct2Policy._output_action_dim(policy, batch)
        values = torch.zeros(batch['n'], 3, width)
        values[..., 8:] = 100.0  # auxiliary hand slots must not become joint metrics
        return values

    def postprocess(payload):
        values = payload[ACTION]
        assert values.shape[-1] == 15
        assert payload['anchor_state'].shape[-1] == 15
        calls.append(values.clone())
        return values[..., :8] + 1.0

    policy.predict_action_chunk = predict
    adapter = MolmoAct2Adapter(policy, None, postprocess, torch.device('cpu'), SimpleNamespace(policy=config))
    adapter._make_batch = lambda *args, **kwargs: {'n': 1}
    n = 3 if batched else 1
    if batched:
        raw, norm = adapter._decode_batch({'n': n}, torch.zeros(n, 8), inference_action_mode=mode)
    else:
        raw, norm, _ = adapter.predict_action_chunk_in_mode(
            {OBS_STATE: torch.zeros(1, 8)}, 'move the sock', inference_action_mode=mode)
    assert calls and calls[0].shape == (n, 3, 15)
    assert norm.shape[-1] == raw.shape[-1] == 8
    assert torch.equal(norm, torch.zeros_like(norm))
    assert torch.equal(raw, torch.ones_like(raw))


def test_normalized_demo_and_predictions_share_joint_width(monkeypatch):
    from lerobot.probes.adapters import molmoact2
    config = SimpleNamespace(hand_block=True,
        input_features={}, output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(15,))})
    adapter = MolmoAct2Adapter(None, None, None, torch.device('cpu'), SimpleNamespace(policy=config))

    def normalizer(transition):
        assert transition[TransitionKey.ACTION].shape == (1, 3, 15)
        return transition

    monkeypatch.setattr(molmoact2, 'find_normalizer_step', lambda _: normalizer)
    normalized = adapter.normalize_gt_actions(torch.ones(3, 8), torch.zeros(8))
    assert normalized.shape == (3, 8)
    assert torch.equal(normalized, torch.ones(3, 8))


@pytest.mark.parametrize('policy_width', [8, 15])
@pytest.mark.parametrize('native_width', [7, 8])
@pytest.mark.parametrize('mask_source', ['extra', 'prepared', 'absent'])
def test_representation_capture_packs_raw_mask_before_policy_width(
    policy_width, native_width, mask_source,
):
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2UnifiedLayoutProcessorStep

    config = SimpleNamespace(
        chunk_size=3, num_flow_timesteps=2, input_features={},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(policy_width,))},
    )
    encoder = torch.nn.Linear(4, 4)
    expert = torch.nn.Module()
    expert.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    backbone = SimpleNamespace(
        transformer=SimpleNamespace(blocks=[encoder]), _require_action_expert=lambda: expert,
    )
    unified = MolmoAct2UnifiedLayoutProcessorStep(
        state_dim=18 if policy_width == 15 else 8, action_dim=policy_width,
        native_action_dims=[native_width], stats_index_key='action_layout_id',
    )
    mask = torch.arange(8).unsqueeze(0) >= native_width
    obs = {OBS_STATE: torch.zeros(1, 8)}
    extra = {'action_layout_id': torch.tensor([0])}
    if mask_source == 'extra':
        extra['action_dim_is_pad'] = mask
    elif mask_source == 'prepared':
        obs['probe_complementary.action_dim_is_pad'] = mask

    def preprocess(flat):
        assert not any(str(k).startswith('probe_complementary.') for k in flat)
        packed = unified({
            TransitionKey.OBSERVATION: {OBS_STATE: flat[OBS_STATE]},
            TransitionKey.ACTION: flat[ACTION],
            TransitionKey.COMPLEMENTARY_DATA: flat[TransitionKey.COMPLEMENTARY_DATA],
        })
        packed_mask = packed[TransitionKey.COMPLEMENTARY_DATA]['action_dim_is_pad']
        assert packed_mask.shape == (1, policy_width)
        assert (~packed_mask).sum() == native_width
        # The tokenizer pack step pads again to the expert's max width.
        return {ACTION: torch.nn.functional.pad(packed[TransitionKey.ACTION], (0, 32 - policy_width))}

    def forward(*, batch, noise, timesteps, **kwargs):
        assert batch[ACTION].shape == (1, 3, 32)
        assert noise.shape == (1, 2, 3, 32)
        assert timesteps.shape == (1, 2) and not timesteps.any()
        encoder(torch.ones(1, 2, 4))
        expert.blocks[0](torch.ones(1, 3, 4))

    policy = SimpleNamespace(
        config=config, _backbone=lambda: backbone, _model_inputs=lambda batch: {},
        _compute_flow_matching_loss_joint_per_layer=forward,
    )
    adapter = MolmoAct2Adapter(policy, preprocess, None, torch.device('cpu'), SimpleNamespace(policy=config))
    adapter._prompt_token_groups = lambda *args: {'state': [0, 1], 'missing': []}
    result = adapter.capture_layer_representations(obs, 'move', extra_complementary=extra)
    assert result['encoder']['state'].shape == (1, 4)
    assert result['encoder']['missing'] is None
    assert result['action_expert']['action'].shape == (1, 4)
    assert not encoder._forward_hooks and not expert.blocks[0]._forward_hooks
    assert torch.equal(mask, torch.arange(8).unsqueeze(0) >= native_width)
