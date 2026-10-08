"""Analytic checks for inverse direction, discrete refinement and forward restarts."""
from types import SimpleNamespace

import pytest
import torch

from lerobot.probes.flow_roundtrip import (
    experiment_grid, forward_path, inverse_path, reconstruction_metrics,
)


def test_constant_field_roundtrips_from_every_time():
    target = torch.tensor([[[2.0, -3.0]]])
    velocity = lambda x, t: torch.ones_like(x) * 2
    inverse = inverse_path(target, velocity, 20)
    torch.testing.assert_close(inverse[0], target - 2)
    for k in (0, 5, 10, 15, 20):
        path = forward_path(inverse[k], velocity, 20, k)
        assert len(path) == 21 - k
        torch.testing.assert_close(path[-1], target)
        torch.testing.assert_close(path, inverse[k:])


def test_refinement_inverts_discrete_map_not_naive_reverse_ode():
    target = torch.tensor([[[2.0]]])
    velocity = lambda x, t: 2 * x + t
    naive = inverse_path(target, velocity, 10)
    refined = inverse_path(target, velocity, 10, refine=12)
    assert abs((forward_path(naive[0], velocity, 10)[-1] - target).item()) > .1
    torch.testing.assert_close(forward_path(refined[0], velocity, 10)[-1], target, atol=2e-6, rtol=0)
    # Analytic discrete inverse of y_next = 1.2*y + (k/10)/10.
    x = target.clone()
    for k in reversed(range(10)):
        x = (x - (k / 10) / 10) / 1.2
    torch.testing.assert_close(refined[0], x)


def test_forward_native_dtype_matches_deployed_euler_arithmetic():
    start = torch.tensor([[[.3, -.7]]], dtype=torch.bfloat16)
    velocity = lambda x, t: x * .4 + torch.tensor(t, dtype=x.dtype)
    actual = forward_path(start, velocity, 20)
    expected = start.clone()
    for k in range(20):
        expected = expected + (1.0 / 20) * velocity(expected, k / 20)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual[-1], expected)
    fp32 = forward_path(start.float(), velocity, 20)
    assert not torch.equal(actual[-1].float(), fp32[-1])


def test_metrics_ignore_padded_joints_and_reject_nonfinite_valid_values():
    target = torch.zeros(1, 2, 3)
    recon = torch.tensor([[[3., 4., float('nan')], [3., 4., 1000.]]])
    valid = torch.tensor([True, True, False])
    metrics = reconstruction_metrics(recon, target, valid)
    assert metrics == pytest.approx({'mse': 12.5, 'rms': (12.5)**.5, 'mae': 3.5, 'max_abs': 4.})
    recon[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='non-finite'):
        reconstruction_metrics(recon, target, valid)


@pytest.mark.parametrize('steps,times', [('0', '0,1'), ('10', '0,0.25'), ('20', '-1,0'), ('20', '0.5,1')])
def test_invalid_experiment_grid_is_rejected(steps, times):
    with pytest.raises(ValueError):
        experiment_grid(SimpleNamespace(inv_roundtrip_steps=steps, inv_roundtrip_refinements='0,4', inv_roundtrip_times=times))


def test_capture_and_report_preserve_paths_and_production_check(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager

    import numpy as np

    from lerobot.probes import utils
    from lerobot.probes.flow_roundtrip import run_experiment

    monkeypatch.setattr(utils, 'probe_image_stride', lambda cfg: 1)
    monkeypatch.setattr(utils, 'sample_episodes_evenly', lambda *args: [(3, 6, 9)])
    monkeypatch.setattr(utils, 'probe_frame_inputs', lambda *args, **kwargs: {})
    calls = []

    @contextmanager
    def prepare(adapter, frame):
        target = torch.tensor([[[2., -3., 0.]]])
        valid = torch.tensor([True, True, False])

        def velocity(x, t, native=False):
            return torch.tensor([[[.5, .5, 0.]]], dtype=x.dtype)

        def deployed(noise, n):
            calls.append(n)
            return forward_path(noise, velocity, n)[-1]

        yield target, valid, velocity, torch.float32, deployed

    own = SimpleNamespace(inv_roundtrip_steps='4', inv_roundtrip_refinements='0,4',
                          inv_roundtrip_times='0,0.5,1', inv_out=str(tmp_path),
                          inv_val_frames_per_episode=1, inv_val_max_episodes=1, inv_seed=42)
    adapter = SimpleNamespace(chunk_size=1, _policy=SimpleNamespace(
        _rtc_enabled=lambda: False, config=SimpleNamespace(mask_action_dim_padding=True)))
    run_experiment(adapter, None, None, own, prepare, {'checkpoint': 'analytic-test'})
    assert len(calls) == 4  # both target types and refinement settings
    data = json.loads((tmp_path / 'roundtrip.json').read_text())
    assert len(data['records']) == 12
    assert max(r['rms'] for r in data['records']) < 1e-6
    assert (tmp_path / 'roundtrip.html').is_file()
    assert (tmp_path / 'roundtrip.png').is_file()
    with np.load(tmp_path / 'trajectories_0000.npz') as arrays:
        assert arrays['demo_n4_r0_inverse'].shape == (5, 1, 1, 3)
        assert arrays['demo_n4_r0_forward_k2'].shape == (3, 1, 1, 3)


def test_production_replay_restores_consumed_modalities_under_same_autocast():
    """A second prefix must not silently drop depth/state/history from the batch."""
    import importlib.util
    from pathlib import Path

    from lerobot.utils.constants import ACTION

    script = Path(__file__).resolve().parents[3] / 'migration/flow_inversion_2026-09-28/flow_inversion.py'
    spec = importlib.util.spec_from_file_location('flow_inversion_regression', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    batch = {ACTION: torch.zeros(1, 2, 3), 'modalities': torch.tensor(2.)}
    contexts, autocast_states = [], []

    class Expert:
        def parameters(self):
            return iter([torch.zeros(1, dtype=torch.bfloat16)])

        def prepare_context(self, **kwargs):
            return kwargs['encoder_kv_states']

        def prepare_modulation_cache(self, times):
            return [SimpleNamespace(conditioning=t) for t in times]

        def forward_with_context(self, x, t, *, context, modulation):
            return torch.ones_like(x) * context

    expert = Expert()
    backbone = SimpleNamespace(
        _extract_kv_states=lambda kv: kv,
        _depth_gate_from_condition=lambda **kwargs: (None, None),
        _apply_depth_gate_to_layer_kv_states=lambda kv, *args: kv,
    )

    class Policy:
        config = SimpleNamespace(dtype='bfloat16', mask_action_dim_padding=True)
        stash = None

        def _backbone(self):
            return backbone

        def _action_expert(self):
            return expert

        def _model_inputs(self, processed):
            assert processed is batch
            self.stash = processed['modalities']
            autocast_states.append(torch.is_autocast_enabled('cpu'))
            return {}

        def _run_prefix_backbone(self, inputs):
            context, self.stash = self.stash, None
            contexts.append(context)
            assert context is not None, 'prefix lost consume-once modalities'
            return SimpleNamespace(past_key_values=context)

        def _mask_action_dim_tensor(self, x, mask):
            return x

        def _encoder_attention_mask_for_action_expert(self, **kwargs):
            return None

        def _generate_actions_from_inputs_with_rtc(self, *, model_inputs, noise, **kwargs):
            context = self._run_prefix_backbone(model_inputs).past_key_values
            return noise + context

    adapter = SimpleNamespace(_policy=Policy(), _device=torch.device('cpu'), action_dim=3,
                              chunk_size=2, _preprocessor=None, _make_batch=lambda *args, **kwargs: batch)
    frame = dict(gt_actions=torch.zeros(2, 3), obs={}, task='', subtask='', metadata={})
    with module.prepared_frame_flow(adapter, frame) as (target, valid, velocity, dtype, deployed):
        for _ in range(2):
            torch.testing.assert_close(deployed(target, 20), target + 2)
        torch.testing.assert_close(velocity(target, 0), torch.full_like(target, 2))
    assert len(contexts) == 3
    assert autocast_states == [True, True, True]


def test_mse_summary_averages_squared_errors_not_rms():
    from lerobot.probes.flow_roundtrip_report import mse_summary

    result = mse_summary([{'rms': 1.}, {'rms': 3.}])
    assert result['mean_mse'] == 5.  # mean([1**2, 3**2]), not mean([1, 3])**2 = 4
    result = mse_summary([{'rms': 1., 'mse': 2.}, {'rms': 3., 'mse': 8.}])
    assert result['mean_mse'] == 5.  # explicitly recorded MSE takes precedence
