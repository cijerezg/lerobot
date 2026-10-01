"""Check transition reconstruction without running a critic."""
import pytest

from lerobot.probes.critic_gradient_view import transition_metrics


def pair(**changes):
    row = dict(source='train', episode=0, global_idx=0, segment_end=89,
               task='sort', subtask='grasp shirt', metadata={'quality': 4},
               images=[{'camera': 'external_0'}], value=-1.0)
    return row, dict(row, global_idx=30, value=-0.8, **changes)


def calculate(a, b, reward=-1/12, done=False):
    return transition_metrics(a, b, reward, done, chunk=30, discount=.97, v_min=-2, v_max=0)


def test_signed_delta_and_discounted_advantage():
    a, b = pair()
    result = calculate(a, b)
    assert result['delta_value'] == pytest.approx(.2)
    assert result['advantage'] == pytest.approx(-1/12 + .97*(-.8) + 1)
    assert not result['target_clipped']


@pytest.mark.parametrize('change', [dict(subtask='release shirt'), dict(metadata={'quality': 3}),
                                   dict(episode=1), dict(source='other'), dict(segment_end=119)])
def test_never_pair_changed_conditioning(change):
    a, b = pair(**change)
    result = calculate(a, b)
    assert result['advantage'] is None and result['delta_value'] is None
    assert result['transition_status'] == 'conditioning_changes'


def test_terminals_need_no_next_forward_and_target_clips():
    a, _ = pair()
    result = calculate(a, None, reward=-5/12, done=True)
    assert result['advantage'] == pytest.approx(1-5/12)
    assert result['delta_value'] is None and result['terminal']
    result = calculate(a, None, reward=-4, done=True)
    assert result['target_clipped'] and result['advantage'] == -1


def test_missing_or_wrong_stride_successor_is_unavailable():
    a, b = pair()
    for nxt in [None, dict(b, global_idx=60)]:
        assert calculate(a, nxt)['advantage'] is None


def test_image_state_norm_excludes_prompt_and_depth_placeholders():
    from lerobot.probes.critic_gradient_view import image_state_gradient_fields
    gradient = dict(state_format='discrete', norm=1000, raw_depth_consumed=False,
                    groups={'img_external_0': {'norm': 3}, 'img_wrist_0': {'norm': 4},
                            'img_external_1': {'norm': None}, 'depth_placeholders': {'norm': 100},
                            'task': {'norm': 200}, 'state': {'norm': 300}},
                    tokens=[{'group': 'state', 'text': '<state_12>', 'norm': 12},
                            {'group': 'state', 'text': '<state_128>', 'norm': 0},
                            {'group': 'state', 'text': ' The', 'norm': 100},
                            {'group': 'state', 'text': '<state_start>', 'norm': 100},
                            {'group': 'state', 'text': '<state_end>', 'norm': 100},
                            {'group': 'state', 'text': '.', 'norm': 100},
                            {'group': 'template', 'text': '<state_123>', 'norm': 100}])
    result = image_state_gradient_fields(gradient)
    assert result['image_grad_norm'] == 5
    assert result['state_value_grad_norm'] == 12
    assert result['image_state_grad_norm'] == 13
    assert result['state_value_token_count'] == 2
    gradient['groups']['other_image_patches'] = {'norm': 12}
    assert image_state_gradient_fields(gradient)['image_grad_norm'] == 13
    gradient['tokens'] = []
    with pytest.raises(ValueError, match='no state-value'):
        image_state_gradient_fields(gradient)


def test_non_discrete_state_records_are_rejected():
    from lerobot.probes.critic_gradient_view import image_state_gradient_fields
    gradient = dict(state_format='continuous', norm=1000,
                    groups={'img_external_0': {'norm': 3}},
                    tokens=[{'group': 'state', 'text': '<extra_0>', 'norm': 12}])
    with pytest.raises(ValueError, match='discrete'):
        image_state_gradient_fields(gradient)
    gradient.pop('state_format')
    with pytest.raises(ValueError, match='discrete'):
        image_state_gradient_fields(gradient)


def test_depth_included_only_when_real_measurements_are_consumed():
    from lerobot.probes.critic_gradient_view import image_state_gradient_fields
    gradient = dict(state_format='discrete', raw_depth_consumed=True, norm=1000,
                    groups={'img_external_0': {'norm': 3}, 'depth': {'norm': 12},
                            'depth_placeholders': {'norm': 100}},
                    tokens=[{'group': 'state', 'text': '<state_7>', 'norm': 4},
                            {'group': 'history_placeholders', 'text': '<extra_0>', 'norm': 100}])
    result = image_state_gradient_fields(gradient)
    assert result['image_state_grad_norm'] == 5
    assert result['observation_grad_norm'] == 13
    assert result['depth_grad_norm'] == 12
    gradient['raw_depth_consumed'] = False
    assert image_state_gradient_fields(gradient)['observation_grad_norm'] == 5
    gradient['raw_depth_consumed'] = True
    gradient['groups'].pop('depth')
    with pytest.raises(ValueError, match='missing its gradient group'):
        image_state_gradient_fields(gradient)
