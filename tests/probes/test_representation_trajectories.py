"""Behavioral checks for temporal geometry, sampling, and the replayable capture."""
import json
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.train import ProbeConfig
from lerobot.probes import representations as rep
from lerobot.probes.representation_trajectories import (
    LAG_EDGES_S, compact, cross_episode_matches, episode_center, episode_cosine_matrices,
    lag_profile, project_pca, robot_center, temporal_metrics,
)


def rows_for(episodes, times=None, robots=None, tasks=None):
    return [dict(row=i, sample=i, episode=ep, episode_label=ep, frame=i,
                 time_s=float(i if times is None else times[i]), robot='A' if robots is None else robots[i],
                 task='pick' if tasks is None else tasks[i], subtask='grasp', holdout=False,
                 text='real', kind='lerobot', source_key='root', index=i)
            for i, ep in enumerate(episodes)]


def test_local_time_window_excludes_self_and_other_episodes_and_handles_zero():
    rows = rows_for(['a', 'a', 'b', 'a', 'a'], [1, 0, 1, 8, 1.5])
    x = np.array([[0., 2], [3., 0], [0., 2], [0., 4], [0., 0]])
    out = temporal_metrics(x, rows, 1.1)
    assert out[0]['local_cosine'] == 0 and out[0]['local_count'] == 1
    assert out[0]['raw_norm'] == 2 and out[0]['previous_dt_s'] == 1
    assert out[0]['step_l2'] == pytest.approx(np.sqrt(13))
    assert out[1]['previous_cosine'] is None  # chronological predecessor, not input order
    assert out[2]['local_cosine'] is None  # isolated episode
    assert out[3]['local_cosine'] is None  # temporal gap, not adjacent list position
    assert out[4]['local_cosine'] is None  # zero vector isn't a direction


def test_matches_exclude_source_episode_and_split_robot_task_in_original_space():
    rows = rows_for(['a', 'a', 'b', 'c', 'd', 'e', 'z'], [0]*7,
                    ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
                    ['pick', 'pick', 'pick', 'place', 'pick', 'place', 'place'])
    x = np.array([[1., 0, 0, 0], [1., 0, 0, 0], [2., 0, 0, 0],
                  [1., 0, 0, 10], [0., 1, 0, 0], [-1., 0, 0, 0], [0., 0, 0, 0]])
    matches, background = cross_episode_matches(x, rows, 1, block_size=2)
    m = matches[0]
    assert m['same_robot_same_task']['target'] == 2
    assert m['same_robot_same_task']['raw_norm_ratio'] == 2
    # Same first 3 coordinates do not imply similarity in the original space.
    assert m['same_robot_different_task']['cosine'] == pytest.approx(1 / np.sqrt(101))
    assert m['different_robot_same_task']['target'] == 4
    assert m['different_robot_different_task']['target'] == 5
    assert m['different_robot_different_task']['cosine'] == -1
    assert all(v is None for v in matches[6].values())
    # Background is the mean over every valid frame of other episodes, not just the best.
    assert background[0]['background_same_robot'] == pytest.approx((1 + 1 / np.sqrt(101)) / 2)
    assert background[0]['background_other_robot'] == pytest.approx(-0.5)
    assert background[6] == {'background_same_robot': None, 'background_other_robot': None}
    assert (matches, background) == cross_episode_matches(x, rows, 1, block_size=128)


def test_target_neighborhood_support_uses_matched_episode_time_and_task():
    rows = rows_for(['a', 'b', 'b', 'b', 'b'], [0, 0, .5, 9, .8],
                    tasks=['one', 'two', 'two', 'two', 'three'])
    x = np.array([[1., 0], [1., 0], [0., 1], [1., 0], [-1., 0]])
    match = cross_episode_matches(x, rows, 1)[0][0]['same_robot_different_task']
    assert match['target'] == 1
    assert match['target_window_count'] == 2
    assert match['target_window_cosine'] == .5
    assert match['candidate_count'] == 4 and match['candidate_episodes'] == 1


def test_training_fits_centering_and_pca_without_holdout_leakage():
    rows = rows_for(['a']*4 + ['b']*2, robots=['A']*4 + ['B']*2)
    rows[3]['holdout'] = True
    rows[4]['holdout'] = rows[5]['holdout'] = True
    x = np.array([[1., 0], [3., 0], [5., 0], [1000., 3], [10., 4], [12., 6]])
    y, means, provenance = robot_center(x, rows)
    np.testing.assert_allclose(means['A'], [3, 0])
    assert y[3, 0] == 997
    assert provenance['B']['fit'].startswith('all')
    coords, pca = project_pca(x, rows)
    assert coords.shape == (6, 3) and not pca['fallback']
    np.testing.assert_allclose(pca['mean'], [3, 0])
    np.testing.assert_allclose(pca['variance'], [1, 0])
    assert set(pca['fit_rows']) == {0, 1, 2}
    changed = x.copy(); changed[3:] *= 100
    np.testing.assert_allclose(project_pca(changed, rows)[1]['components'], pca['components'])


def test_degenerate_pca_is_finite_and_zero_cosines_are_missing():
    rows = rows_for(['a', 'b'])
    x = np.ones((2, 1))
    coords, pca = project_pca(x, rows)
    assert coords.shape == (2, 3) and not coords.any()
    assert not pca['variance'].any()
    y = robot_center(x, rows)[0]
    assert all(v is None for v in cross_episode_matches(y, rows, 1)[0][0].values())


def test_episode_centering_matrices_and_lag_profile_follow_time_not_input_order():
    rows = rows_for(['a', 'a', 'b', 'a', 'a'], [2, 0, 5, 1, 3.5])
    x = np.array([[1., 0], [0., 1], [1., 1], [1., 0], [0., 0]])
    y = episode_center(x, rows)
    np.testing.assert_allclose(y[[0, 1, 3, 4]].mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(y[2], 0)  # a single-frame episode centers to nothing
    matrices = episode_cosine_matrices(x, rows)
    a = matrices['a']
    assert a['ids'].tolist() == [1, 3, 0, 4] and a['times'].tolist() == [0, 1, 2, 3.5]
    np.testing.assert_allclose(a['cosine'][:3, :3], [[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    assert np.isnan(a['cosine'][3]).all() and np.isnan(a['cosine'][:, 3]).all()  # zero vector
    profile = lag_profile(a['cosine'], a['times'], edges=(0, 1, 2))
    # Pairs (0,1)->gap 1 bin 0, (0,2)->2 bin 1, (1,2)->1 bin 0; frame 3 pairs are invalid.
    assert [(p['lag_lo_s'], p['lag_hi_s'], p['pairs']) for p in profile] == [(0, 1, 2), (1, 2, 1), (2, None, 0)]
    assert profile[0]['mean_cosine'] == .5 and profile[1]['mean_cosine'] == 0 and profile[2]['mean_cosine'] is None
    assert lag_profile(a['cosine'], a['times'])[-1]['lag_lo_s'] == LAG_EDGES_S[-1]
    assert compact(np.array([[1.23456, np.nan], [np.inf, -0.00004]])) == [[1.2346, None], [None, -0.0]]


def test_sampling_balances_tasks_robots_and_preserves_episode_ends():
    candidates = []
    for robot in ('A', 'B'):
        for task in ('pick', 'place'):
            for ep in range(4):
                rows = rows_for([f'{robot}/{task}/{ep}']*10, list(range(10)), [robot]*10, [task]*10)
                candidates.append(rows)
    selected = rep.select_sequences(candidates, 2, 8, 20, 42)
    assert selected == rep.select_sequences(candidates, 2, 8, 20, 42)
    assert len(selected) == 20
    assert Counter(r['robot'] for r in selected) == {'A': 10, 'B': 10}
    assert Counter(r['task'] for r in selected) == {'pick': 10, 'place': 10}
    for episode in {r['episode'] for r in selected}:
        times = [r['time_s'] for r in selected if r['episode'] == episode]
        assert times == sorted(set(times)) and times[0] == 0 and times[-1] == 9
    with pytest.raises(ValueError, match='multiple sources'):
        rep.select_sequences(candidates + candidates[:1], 2, 8, 20, 42)
    with pytest.raises(ValueError, match='>= 2'):
        rep.select_sequences(candidates, 2, 1, 20, 42)


def fake_capture(tmp_path, monkeypatch):
    calls = []
    samples = rows_for(['a', 'a', 'b', 'b'], [0, 1, 0, 1], tasks=['pick', 'pick', 'place', 'place'])
    def inputs(dataset, cfg, sample, chunk):
        return dict(obs={'index': sample['index']}, task=sample['task'], subtask='grasp', metadata={}, extra={})
    class Adapter:
        chunk_size = 30
        def capture_layer_representations(self, obs, task, **kwargs):
            calls.append((task, kwargs))
            v = torch.arange(12, dtype=torch.float32).reshape(3, 4) + obs['index'] + 1
            native = dict(encoder={'state': None if obs['index'] == 0 else v + 10}, action_expert={'action': v})
            return dict(encoder={'state': v + 100}, action_expert={'action': v + 100},
                        native_views={'attention_input': native})
    p = ProbeConfig(repr_layers='0,2', repr_groups='action,state', repr_local_window_s=1.1)
    cfg = SimpleNamespace(probe_parameters=p, policy=SimpleNamespace(pretrained_path='test-checkpoint'))
    monkeypatch.setattr(rep, '_rebot_inputs', inputs)
    monkeypatch.setattr(rep, '_save_thumbs', lambda *args: None)
    rep.collect(Adapter(), cfg, samples, {'root': None}, {}, tmp_path / 'cache')
    return cfg, calls


def test_capture_preserves_native_float32_absence_fixed_noise_and_text_controls(tmp_path, monkeypatch):
    cfg, calls = fake_capture(tmp_path, monkeypatch)
    meta, arrays, present = rep.load_cache(tmp_path / 'cache')
    assert len(meta['rows']) == 8
    assert calls[1][0] == rep.NEUTRAL_TASK and calls[1][1]['subtask'] == rep.NEUTRAL_SUBTASK
    assert {c[1]['noise_seed'] for c in calls} == {cfg.probe_parameters.random_seed}
    assert all(c[1]['capture_native_views'] for c in calls)
    assert arrays['action'].dtype == np.float32
    np.testing.assert_allclose(arrays['action'][0, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(arrays['action'][0, 1], [9, 10, 11, 12])
    assert present['state'].tolist() == [False, False, True, True, True, True, True, True]


def test_model_free_replot_writes_linked_report_metrics_basis_and_manifest(tmp_path, monkeypatch):
    cfg, _ = fake_capture(tmp_path, monkeypatch)
    cfg.probe_parameters.mode = 'plot'
    summary = rep.run(None, None, cfg, tmp_path)
    assert summary['frames'] == 4 and summary['episodes'] == 2
    html = (tmp_path / 'trajectories.html').read_text()
    assert 'plotly' in html and 'Target-window cosine' in html
    assert '<script id="data"' in html
    assert 'umap' not in html.lower()
    # The page stays light: per-panel data and previews are loaded from disk on demand.
    assert 'data:image/jpeg' not in html and '"cosine"' not in html
    light = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert len(light['panels']) == 24 and light['panels'][0]['file'].startswith('panels/')
    assert light['thumb_dir'] == 'cache/thumbs' and light['lag_edges_s'] == list(LAG_EDGES_S)
    sidecars = sorted((tmp_path / 'panels').glob('*.js'))
    assert len(sidecars) == 24
    text = (tmp_path / 'panels' / 'action.L0.real.episode_centered.js').read_text()
    assert text.startswith('window.__trajectoryPanel("action.L0.real.episode_centered",')
    data = json.loads(text[text.index(',') + 1:-2])
    episode = data['episodes']['a']
    assert episode['ids'] == [0, 1] and [row[i] for i, row in enumerate(episode['cosine'])] == [1, 1]
    assert len(episode['lags']) == len(LAG_EDGES_S)
    assert set(data['temporal']) >= {'local_cosine', 'background_same_robot', 'raw_norm'}
    assert data['matches']['same_robot_different_task']['target'][0] in (2, 3)
    assert data['matches']['same_robot_same_task']['target'] == [None] * 4
    assert (tmp_path / 'temporal.csv').stat().st_size > 100
    assert (tmp_path / 'matches.csv').stat().st_size > 100
    assert 'lag_lo_s' in (tmp_path / 'lags.csv').read_text()
    assert len(list((tmp_path / 'pca').glob('*.npz'))) == 24
    index = json.loads((tmp_path / 'index.json').read_text())
    assert index['panels'][0]['file'] == 'trajectories.html'
    assert any(p['file'] == 'lags.csv' for p in index['panels'])
    cfg.probe_parameters.repr_view = 'block_output'
    with pytest.raises(ValueError, match='differ from cache'):
        rep.run(None, None, cfg, tmp_path)


def test_legacy_or_incomplete_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='whole-sequence'):
        rep.load_cache(tmp_path)
    (tmp_path / 'meta.json').write_text('{"protocol":"legacy"}')
    with pytest.raises(ValueError, match='protocol'):
        rep.load_cache(tmp_path)
