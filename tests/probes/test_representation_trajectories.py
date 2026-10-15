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
    MATCH_TOP_K, compact, cross_episode_matches, episode_center, episode_cosine_matrices,
    full_cosine, pack_cosine, robot_center, temporal_metrics, unpack_cosine,
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
    m = {kind: ranked[0] for kind, ranked in matches[0].items()}
    assert m['same_robot_same_task']['target'] == 2
    assert m['same_robot_same_task']['raw_norm_ratio'] == 2
    # Same first 3 coordinates do not imply similarity in the original space.
    assert m['same_robot_different_task']['cosine'] == pytest.approx(1 / np.sqrt(101))
    assert m['different_robot_same_task']['target'] == 4
    assert m['different_robot_different_task']['target'] == 5
    assert m['different_robot_different_task']['cosine'] == -1
    assert all(v == [] for v in matches[6].values())
    # Background is the mean over every valid frame of other episodes, not just the best.
    assert background[0]['background_same_robot'] == pytest.approx((1 + 1 / np.sqrt(101)) / 2)
    assert background[0]['background_other_robot'] == pytest.approx(-0.5)
    assert background[0]['background_distance_other_robot'] == pytest.approx((np.sqrt(2) + 2) / 2)
    assert background[0]['background_lognorm_same_robot'] == pytest.approx((1 + np.log2(np.sqrt(101))) / 2)
    assert all(v is None for v in background[6].values()) and len(background[6]) == 6
    assert (matches, background) == cross_episode_matches(x, rows, 1, block_size=128)


def test_target_neighborhood_support_uses_matched_episode_time_and_task():
    rows = rows_for(['a', 'b', 'b', 'b', 'b'], [0, 0, .5, 9, .8],
                    tasks=['one', 'two', 'two', 'two', 'three'])
    x = np.array([[1., 0], [1., 0], [0., 1], [1., 0], [-1., 0]])
    match = cross_episode_matches(x, rows, 1)[0][0]['same_robot_different_task'][0]
    assert match['target'] == 1
    assert match['target_window_count'] == 2
    assert match['target_window_cosine'] == .5
    assert match['candidate_count'] == 4 and match['candidate_episodes'] == 1


def test_ranked_matches_keep_one_frame_per_episode_best_first():
    # Episode b has two frames close to the anchor; c, d, e one each; z is a zero vector.
    rows = rows_for(['a', 'b', 'b', 'c', 'd', 'e', 'z'], [0] * 7)
    x = np.array([[1., 0], [.9, .1], [.95, .05], [.5, .5], [.8, .2], [0., 1], [0., 0]])
    ranked = cross_episode_matches(x, rows, 1, top_k=3)[0][0]['same_robot_same_task']
    assert [rows[m['target']]['episode'] for m in ranked] == ['b', 'd', 'c']
    assert ranked[0]['target'] == 2  # the better of episode b's two frames
    assert [m['cosine'] for m in ranked] == sorted((m['cosine'] for m in ranked), reverse=True)
    assert all(m['candidate_count'] == 5 and m['candidate_episodes'] == 4 for m in ranked)
    assert len(cross_episode_matches(x, rows, 1)[0][0]['same_robot_same_task']) == min(MATCH_TOP_K, 4)


def test_training_fits_centering_without_holdout_leakage():
    rows = rows_for(['a']*4 + ['b']*2, robots=['A']*4 + ['B']*2)
    rows[3]['holdout'] = True
    rows[4]['holdout'] = rows[5]['holdout'] = True
    x = np.array([[1., 0], [3., 0], [5., 0], [1000., 3], [10., 4], [12., 6]])
    y, means, provenance = robot_center(x, rows)
    np.testing.assert_allclose(means['A'], [3, 0])
    assert y[3, 0] == 997
    assert provenance['B']['fit'].startswith('all')


def test_zero_cosines_are_missing():
    rows = rows_for(['a', 'b'])
    y = robot_center(np.ones((2, 1)), rows)[0]
    assert all(v == [] for v in cross_episode_matches(y, rows, 1)[0][0].values())


def test_episode_centering_and_matrices_follow_time_not_input_order():
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
    assert compact(np.array([[1.23456, np.nan], [np.inf, -0.00004]])) == [[1.2346, None], [None, -0.0]]
    full = full_cosine(x)
    np.testing.assert_allclose(full[np.ix_(a['ids'], a['ids'])], a['cosine'])
    assert episode_cosine_matrices(x, rows, full)['a']['cosine'] is not None
    back = unpack_cosine(pack_cosine(full), len(x))
    np.testing.assert_allclose(back[:4, :4][~np.isnan(full[:4, :4])], full[:4, :4][~np.isnan(full[:4, :4])], atol=4e-5)
    assert np.isnan(back[4]).all() and np.isnan(back[:, 4]).all()
    np.testing.assert_allclose(unpack_cosine(pack_cosine([[-1.]]), 1), [[-1.]])
    np.testing.assert_allclose(unpack_cosine(pack_cosine([[1.]]), 1), [[1.]])


def test_stride_picks_nearest_frame_per_step_without_interpolation():
    assert rep.stride_picks([0, .4, 1.1, 2.2, 2.9, 4.3], 1.0) == [0, 2, 3, 4, 5]  # 0, 1.1, 2.2, 2.9, 4.3
    assert rep.stride_picks([0, .4, 1.1, 2.2, 2.9, 4.3], 2.0) == [0, 3, 5]
    assert rep.stride_picks([0, 10], 2.0) == [0, 1]  # sparse data gives fewer frames, never invented ones
    assert rep.stride_picks([3, 1, 2], 1.0) == [1, 2, 0]  # positions in the given list, chronological
    assert rep.stride_picks([], 1.0) == [] and rep.stride_picks([5.], 1.0) == [0]


def test_sampling_balances_tasks_robots_and_keeps_whole_episodes_at_the_stride():
    candidates = []
    for robot in ('A', 'B'):
        for task in ('pick', 'place'):
            for ep in range(4):
                rows = rows_for([f'{robot}/{task}/{ep}']*10, list(range(10)), [robot]*10, [task]*10)
                candidates.append(rows)
    selected = rep.select_sequences(candidates, 2, 45, 42)
    assert selected == rep.select_sequences(candidates, 2, 45, 42)
    assert len(selected) == 40  # whole episodes only: a fifth would not fit under 45
    assert Counter(r['robot'] for r in selected) == {'A': 20, 'B': 20}
    assert Counter(r['task'] for r in selected) == {'pick': 20, 'place': 20}
    for episode in {r['episode'] for r in selected}:
        times = [r['time_s'] for r in selected if r['episode'] == episode]
        assert times == list(range(10))  # never thinned
    # Robots are visited in sorted order: an episode that does not fit is skipped, never
    # thinned, and a later shorter one that fits is still taken.
    long_a, long_b = candidates[0], candidates[8]
    short = rows_for(['C/pick/short']*3, [0, 1, 2], ['C']*3, ['pick']*3)
    def pick(budget, **kw):
        return {r['episode'] for r in rep.select_sequences([long_a, long_b, short], None, budget, 42, **kw)}
    assert pick(12, min_frames_per_episode=2) == {long_a[0]['episode']}
    assert pick(14, min_frames_per_episode=2) == {long_a[0]['episode'], 'C/pick/short'}
    assert pick(23, min_frames_per_episode=2) == {long_a[0]['episode'], long_b[0]['episode'], 'C/pick/short'}
    # Episodes below the minimum are left out before balancing.
    assert pick(23, min_frames_per_episode=5) == {long_a[0]['episode'], long_b[0]['episode']}
    # Held-out episodes have their own cap and go first in the robot's queue, so
    # max_episodes counts training episodes only.
    held = []
    for ep in range(3):
        rows = rows_for([f'A/val/{ep}']*10, list(range(10)), ['A']*10, ['pick']*10)
        for r in rows:
            r['holdout'] = True
        held.append(rows)
    chosen = rep.select_sequences(candidates + held, 2, 1000, 42)
    assert Counter((r['robot'], r['holdout']) for r in chosen) == {('A', True): 30, ('A', False): 20, ('B', False): 20}
    chosen = rep.select_sequences(candidates + held, 2, 1000, 42, max_holdout_episodes=1)
    assert Counter((r['robot'], r['holdout']) for r in chosen) == {('A', True): 10, ('A', False): 20, ('B', False): 20}
    tight = {r['episode'] for r in rep.select_sequences(candidates + held, 2, 30, 42) if r['robot'] == 'A'}
    assert tight <= {f'A/val/{ep}' for ep in range(3)}  # val first when the budget is tight
    with pytest.raises(ValueError, match='multiple sources'):
        rep.select_sequences(candidates + candidates[:1], 2, 45, 42)
    with pytest.raises(ValueError, match='>= 2'):
        rep.select_sequences(candidates, 2, 1, 42)
    with pytest.raises(ValueError, match='fits under'):
        rep.select_sequences(candidates, 2, 5, 42)


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
    assert 'plotly' in html and 'Counterparts of the anchor frame' in html
    assert '<script id="data"' in html
    assert 'umap' not in html.lower()
    # The page stays light: per-panel data and previews are loaded from disk on demand.
    assert 'data:image/jpeg' not in html and '"cosine":[' not in html
    light = json.loads(html.split('<script id="data" type="application/json">')[1].split('</script>')[0])
    assert len(light['panels']) == 24 and light['panels'][0]['file'].startswith('panels/')
    assert light['thumb_dir'] == 'cache/thumbs' and light['match_top_k'] == MATCH_TOP_K
    assert light['stride_s'] == cfg.probe_parameters.repr_stride_s
    assert 'variance' not in light['panels'][0] and 'PCA' not in html and 'How local is local' not in html
    sidecars = sorted((tmp_path / 'panels').glob('*.js'))
    assert len(sidecars) == 48 and len(list((tmp_path / 'panels').glob('*.cross.js'))) == 24
    assert light['panels'][0]['cross_file'].endswith('.cross.js')
    cross_text = (tmp_path / 'panels' / 'action.L0.real.raw.cross.js').read_text()
    packed = json.loads(cross_text[cross_text.index(',') + 1:-2])
    full = unpack_cosine(packed, 4)
    np.testing.assert_allclose(np.diag(full), 1, atol=4e-5)
    assert '"cross"' not in (tmp_path / 'panels' / 'action.L0.real.raw.js').read_text()
    text = (tmp_path / 'panels' / 'action.L0.real.episode_centered.js').read_text()
    assert text.startswith('window.__trajectoryPanel("action.L0.real.episode_centered",')
    data = json.loads(text[text.index(',') + 1:-2])
    episode = data['episodes']['a']
    assert episode['ids'] == [0, 1] and [row[i] for i, row in enumerate(episode['cosine'])] == [1, 1]
    assert 'lags' not in episode and 'xyz' not in data
    assert set(data['temporal']) >= {'local_cosine', 'background_same_robot', 'raw_norm'}
    # Ranked matches: one list per frame, best first, one frame per other episode.
    assert data['matches']['same_robot_different_task']['target'][0] in ([2], [3])
    assert data['matches']['same_robot_same_task']['target'] == [[]] * 4
    assert (tmp_path / 'temporal.csv').stat().st_size > 100
    assert ',rank,' in (tmp_path / 'matches.csv').read_text()
    assert not (tmp_path / 'lags.csv').exists() and not (tmp_path / 'pca').exists()
    index = json.loads((tmp_path / 'index.json').read_text())
    assert index['panels'][0]['file'] == 'trajectories.html'
    assert not any(p['file'] == 'lags.csv' for p in index['panels'])
    cfg.probe_parameters.repr_cross_matrices = False
    rep.run(None, None, cfg, tmp_path)
    assert not list((tmp_path / 'panels').glob('*.cross.js'))
    assert json.loads((tmp_path / 'summary.json').read_text())['cross_matrices'] is False
    cfg.probe_parameters.repr_view = 'block_output'
    with pytest.raises(ValueError, match='differ from cache'):
        rep.run(None, None, cfg, tmp_path)


def test_legacy_or_incomplete_cache_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='whole-sequence'):
        rep.load_cache(tmp_path)
    (tmp_path / 'meta.json').write_text('{"protocol":"legacy"}')
    with pytest.raises(ValueError, match='protocol'):
        rep.load_cache(tmp_path)
